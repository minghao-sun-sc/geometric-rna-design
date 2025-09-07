# Proposed Fix for Batch Structure Issue

## Problem
The `collate_pairs` function returns a tuple:
```python
(data_batch, y_w, y_l, w, node_mask, gids)
```

But `_unwrap_batch` in the Lightning module incorrectly takes only `batch[0]`, losing all the other data needed for DPO training.

## Solution Options

### Option 1: Fix the batch structure in collate_pairs (RECOMMENDED)
Modify `collate_pairs` to attach all data to the PyG batch object:

```python
def collate_pairs(batch: List[Tuple[Any, ...]]):
    datas, ys_w, ys_l, ws, masks, gids = zip(*batch)
    data_batch = Batch.from_data_list(datas)
    
    # Attach DPO-specific data to the batch
    data_batch.y_w = torch.cat(ys_w, dim=0)
    data_batch.y_l = torch.cat(ys_l, dim=0)
    data_batch.weight = torch.stack(ws)
    
    if masks[0] is not None:
        data_batch.node_mask = torch.cat(masks, dim=0)
    else:
        data_batch.node_mask = torch.ones(data_batch.y_w.shape[0], dtype=torch.float32)
    
    data_batch.gids = list(gids)
    
    return data_batch  # Return single batch object with all attributes
```

### Option 2: Fix _shared_step to handle tuple correctly
Modify `_shared_step` to properly unpack the tuple:

```python
def _shared_step(self, batch: Any, batch_idx: int, *, train: bool) -> Dict[str, torch.Tensor]:
    # Handle the tuple structure from collate_pairs
    if isinstance(batch, (list, tuple)) and len(batch) == 6:
        data_batch, y_w, y_l, w, node_mask, gids = batch
        # Create a combined batch object
        data_batch.y_w = y_w
        data_batch.y_l = y_l
        data_batch.weight = w
        data_batch.node_mask = node_mask if node_mask is not None else torch.ones_like(y_w, dtype=torch.float32)
        data_batch.gids = gids
        batch = data_batch
    else:
        batch = _unwrap_batch(batch)
    
    # Continue with dpo_sft_step...
```

### Option 3: Create a custom batch class
Define a custom batch class that holds all the necessary data:

```python
class DPOBatch:
    def __init__(self, graph_batch, y_w, y_l, weight, node_mask, gids):
        self.graph = graph_batch
        self.y_w = y_w
        self.y_l = y_l
        self.weight = weight
        self.node_mask = node_mask
        self.gids = gids
```

## Recommendation
**Option 1** is the cleanest - it keeps everything in a single PyG batch object that naturally moves through Lightning's pipeline and automatically handles device transfers.