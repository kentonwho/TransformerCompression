# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .config import config
from .model_adapter import LayerAdapter, ModelAdapter
from .model_utils import get_layer0_inputs, get_signals
from .slicing_scheduler import ConfigSlicingScheduler, ConstSlicingScheduler, SlicingScheduler
from .utils import cleanup_memory, map_tensors


def rotate_attention_inputs(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in layer_adapter.get_attention_inputs():
        dtype = W.weight.dtype
        W_ = W.weight.to(device=config.device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def slice_attention_inputs(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    # Slice the WQ, WK and WV matrices of the self-attention layer.
    for W in layer_adapter.get_attention_inputs():
        W.weight.data = W.weight.data[:, :new_embedding_dimension]
        W.in_features = new_embedding_dimension

    layer_adapter.layer.attn_shortcut_Q = nn.Parameter(layer_adapter.layer.attn_shortcut_Q[:new_embedding_dimension, :])

def project_attention_inputs(layer_adapter: LayerAdapter, Q: torch.Tensor, new_embedding_dimension:int) -> None:
    # Project the WQ, WK and WV matrices of the self-attention layer.
    for W in layer_adapter.get_attention_inputs():
        dtype = W.weight.dtype
        W_ = W.weight.to(device=config.device, dtype=torch.float64)
        W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
        W.in_features = new_embedding_dimension
    
    layer_adapter.layer.attn_shortcut_Q = nn.Parameter(layer_adapter.layer.attn_shortcut_Q[:new_embedding_dimension, :])


def rotate_attention_output(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    # Rotate output matrix of the self-attention layer.
    W = layer_adapter.get_attention_output()

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=config.device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def slice_attention_output(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    # Slice output matrix of the self-attention layer.
    W = layer_adapter.get_attention_output()
    W.weight.data = W.weight.data[:new_embedding_dimension, :]
    if W.bias is not None:
        W.bias.data = W.bias.data[:new_embedding_dimension]
    W.out_features = new_embedding_dimension

def project_attention_output(layer_adapter: LayerAdapter, Q: torch.Tensor, new_embedding_dimension: int) -> None:
    # Project output matrix of the self-attention layer.
    W = layer_adapter.get_attention_output()

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=config.device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)
    W.out_features = new_embedding_dimension

def rotate_mlp_input(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    # Rotate the MLP input weights.
    for W in layer_adapter.get_mlp_inputs():
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def slice_mlp_input(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    # Slice the MLP input weights.
    for W in layer_adapter.get_mlp_inputs():
        W.weight.data = W.weight.data[:, :new_embedding_dimension]
        W.in_features = new_embedding_dimension

#keep new_embedding_dimensions for now just for consistency (don't really need it since we can read from Q)
def project_mlp_input(layer_adapter: LayerAdapter, Q: torch.Tensor, new_embedding_dimension: int) -> None:
    # Project the MLP input weights.
    for W in layer_adapter.get_mlp_inputs():
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
        W.in_features = new_embedding_dimension


def rotate_mlp_output(layer_adapter: LayerAdapter, Q: torch.Tensor) -> None:
    # Rotate the MLP output weights and bias.
    W = layer_adapter.get_mlp_output()
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=config.device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def slice_mlp_output(layer_adapter: LayerAdapter, new_embedding_dimension: int) -> None:
    # Slice the MLP output weights and bias.
    W = layer_adapter.get_mlp_output()
    W.weight.data = W.weight.data[:new_embedding_dimension, :]
    if W.bias is not None:
        W.bias.data = W.bias.data[:new_embedding_dimension]
    W.out_features = new_embedding_dimension

def project_mlp_output(layer_adapter: LayerAdapter, Q: torch.Tensor, new_embedding_dimension: int) -> None:
    # Project the MLP output weights and bias.
    W = layer_adapter.get_mlp_output()
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=config.device, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)
    W.out_features = new_embedding_dimension


def rotate_embeddings(model_adapter: ModelAdapter, Q: torch.Tensor) -> None:
    # Rotate the embeddings.
    for W in model_adapter.get_embeddings():
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

    # Run GC and cleanup GPU memory
    cleanup_memory()


def slice_embeddings(model_adapter: ModelAdapter, new_embedding_dimensions: dict[int, int]) -> None:
    # Slice the embeddings.
    for i, W in enumerate(model_adapter.get_embeddings()):
        W.weight.data = W.weight.data[:, : new_embedding_dimensions[i]]
        W.embedding_dim = new_embedding_dimensions[i]

def project_embeddings(model_adapter: ModelAdapter, Q: torch.Tensor, new_embedding_dimensions: dict[int,int]) -> None:
    # Project the embeddings.
    for i, W in enumerate(model_adapter.get_embeddings()):
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
        W.embedding_dim = new_embedding_dimensions[i]

    cleanup_memory()

def rotate_head(model_adapter: ModelAdapter, Q: torch.Tensor) -> None:
    # Rotate the head.
    W = model_adapter.get_lm_head()
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=config.device, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def slice_head(model_adapter: ModelAdapter, new_embedding_dimension: int) -> None:
    # Slice the head.
    lm_head = model_adapter.get_lm_head()
    lm_head.weight.data = lm_head.weight.data[:, :new_embedding_dimension]
    lm_head.in_features = new_embedding_dimension

def project_head(model_adapter: ModelAdapter, Q: torch.Tensor, new_embedding_dimension: int) -> None:
    # Project the head.
    lm_head = model_adapter.get_lm_head()
    dtype = lm_head.weight.data.dtype
    W_ = lm_head.weight.data.to(device=config.device, dtype=torch.float64)
    lm_head.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
    lm_head.in_features = new_embedding_dimension

def rotate_and_slice(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    slicing_scheduler: SlicingScheduler,
    apply_mask: bool = True,
    final_orientation: str = 'pca',
) -> None:
    """
    Rotate and slice a model, with interleaved slicing and PCA calculations
    """
    if model_adapter.parallel_blocks:
        rotate_and_slice_parallel(model_adapter, dataloader, slicing_scheduler, apply_mask, final_orientation)
    else:
        rotate_and_slice_sequential(model_adapter, dataloader, slicing_scheduler, apply_mask, final_orientation)


@torch.no_grad()
def rotate_and_slice_sequential(
    model_adapter: ModelAdapter,
    dataloader: torch.utils.data.DataLoader[torch.Tensor],
    slicing_scheduler: SlicingScheduler,
    apply_mask: bool = True,
    final_orientation: str = 'pca',
) -> None:
    """
    Rotate and slice the provided model, with interleaved slicing and PCA calculations.

    This method works for models where the MLP block is computed after the attention block.
    """
    model_adapter.model.eval()
    dtype = next(iter(model_adapter.model.parameters())).dtype

    inps, args, kwargs, ignore_masks = [], [], [], []
    for batch in dataloader:
        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)
        if apply_mask:
            ignore_masks.append(batch["attention_mask"])

    layers = model_adapter.get_layers()
    slicing_scheduler.setup(hidden_size=model_adapter.hidden_size, layers_num=len(layers), parallel_blocks=False)

    # rotate and slice embeddings
    #eig_val, Q = pca_calc(inps, ignore_masks)
    
    Q = svd_calc(inps, slicing_scheduler.get_embedding_dimensions(), ignore_masks)
    Q = Q.to(device=config.device)

    #KENTON: I don't totally understand this part. I guess they're trying to cheapen the cost of the projection step? 
    #if final_orientation == 'random':
    #    R = random_orthogonal_upper_left(Q.shape[0], slicing_scheduler.get_embedding_dimensions()[0])
    #    Q = Q @ R.to(Q.device)
    #rotate_embeddings(model_adapter, Q)
    #slice_embeddings(model_adapter, slicing_scheduler.get_embedding_dimensions())

    project_embeddings(model_adapter, Q, slicing_scheduler.get_embedding_dimensions())

    logging.info("Rotate and slice layers")
    for idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Rotating and slicing")):
        layer = layer_adapter.layer
        layer.attn_shortcut_Q = nn.Parameter(Q.T.clone().to(dtype=dtype))

        # rotate and slice the attention inputs to match previous layer
        #rotate_attention_inputs(layer_adapter, Q)
        #slice_attention_inputs(layer_adapter, slicing_scheduler.get_attention_input_dimension(idx))

        project_attention_inputs(layer_adapter, Q, slicing_scheduler.get_attention_input_dimension(idx))

        # get signal between attention and mlp, rotate and slice
        for i, inp in enumerate(inps):
            args[i] = layer_adapter.get_updated_args(
                torch.matmul(inp.to(device=config.device), Q.to(dtype=dtype)).cpu(),
                args[i],
            )

        mlp_ln_inputs, _ = get_signals(layer_adapter, args, kwargs)
        #eig_val, Q = pca_calc(mlp_ln_inputs, ignore_masks)
        Q = svd_calc(mlp_ln_inputs, slicing_scheduler.get_attention_output_dimension(idx), ignore_masks)
        Q = Q.to(device=config.device, dtype=torch.float64)

        """ if final_orientation == 'random':
            R = random_orthogonal_upper_left(
                Q.shape[0], slicing_scheduler.get_attention_output_dimension(idx, match_head_dim=False)
            )
            Q = Q @ R.to(Q.device)
 """
        #why is this learnable? 
        layer.attn_shortcut_Q = nn.Parameter(torch.matmul(layer.attn_shortcut_Q, Q.to(dtype=dtype)))
        #rotate_attention_output(layer_adapter, Q)
        #slice_attention_output(
        #    layer_adapter, slicing_scheduler.get_attention_output_dimension(idx, match_head_dim=False)
        #)

        project_attention_output(layer_adapter, Q, slicing_scheduler.get_attention_output_dimension(idx))

        layer.mlp_shortcut_Q = nn.Parameter(
            Q.T.clone().to(dtype=dtype)
        )
        project_mlp_input(layer_adapter, Q, slicing_scheduler.get_mlp_input_dimension(idx))

        # Run GC and cleanup GPU memory
        cleanup_memory()

        # now compute the outputs of the current layer/inputs for the next layer
        # with slicing between Attention and mlp.
        _, inps = get_signals(layer_adapter, args, kwargs)
        #eig_val, Q = pca_calc(inps, ignore_masks)
        Q = svd_calc(inps, slicing_scheduler.get_mlp_output_dimension(idx), ignore_masks)
        """ if final_orientation == 'random':
            R = random_orthogonal_upper_left(Q.shape[0], slicing_scheduler.get_mlp_output_dimension(idx))
            Q = Q @ R.to(Q.device)
 """
        layer.mlp_shortcut_Q = nn.Parameter(torch.matmul(layer.mlp_shortcut_Q, Q.to(dtype=dtype)))

        # optionally slice the mlp/head connection in the last layer
        #rotate_mlp_output(layer_adapter, Q)
        #slice_mlp_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(idx))
        
        project_mlp_output(layer_adapter, Q, slicing_scheduler.get_mlp_output_dimension(idx))
        layer.mlp_shortcut_Q = nn.Parameter(layer.mlp_shortcut_Q)

        layer.to('cpu')

        # Run GC and cleanup GPU memory
        cleanup_memory()

    # rotate and slice head
    #rotate_head(model_adapter, Q)
    if slicing_scheduler.do_slice_head:
        #slice_head(model_adapter, slicing_scheduler.get_head_dimension())
        project_head(model_adapter, Q, slicing_scheduler.get_head_dimension())

    # update model's slicing config
    model_adapter.slicing_conf = slicing_scheduler.slicing_conf.clone()
    logging.info("Rotate and slice layers done")

    @torch.no_grad()
    def rotate_and_slice_parallel(
        model_adapter: ModelAdapter,
        dataloader: torch.utils.data.DataLoader[torch.Tensor],
        slicing_scheduler: SlicingScheduler,
        apply_mask: bool = True,
        final_orientation: str = 'pca',
    ) -> None:
        """
        Rotate and slice a model, with interleaved slicing and PCA calculations

        This version works for models where the MLP block and the attention block are computed in parallel.
        """
        model_adapter.model.eval()
        dtype = next(iter(model_adapter.model.parameters())).dtype

        inps, args, kwargs, ignore_masks = [], [], [], []
        for batch in dataloader:
            inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
            inps.append(inp_batch)
            args.append(args_batch)
            kwargs.append(kwargs_batch)
            if apply_mask:
                ignore_masks.append(batch["attention_mask"])

        layers = model_adapter.get_layers()
        slicing_scheduler.setup(hidden_size=model_adapter.hidden_size, layers_num=len(layers), parallel_blocks=True)

        # rotate and slice embeddings
        Q = svd_calc(inps, slicing_scheduler.get_embedding_dimensions(), ignore_masks)
        Q = Q.to(device=config.device)

        project_embeddings(model_adapter, Q, slicing_scheduler.get_embedding_dimensions())

        logging.info("Rotate and slice layers")
        for idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Rotating and slicing")):
            layer = layer_adapter.layer
            layer.attn_shortcut_Q = nn.Parameter(Q.T.clone().to(dtype=dtype))

            project_attention_inputs(layer_adapter, Q, slicing_scheduler.get_attention_input_dimension(idx))
            project_mlp_input(layer_adapter, Q, slicing_scheduler.get_attention_input_dimension(idx))

            for i, inp in enumerate(inps):
                args[i] = layer_adapter.get_updated_args(
                    torch.matmul(inp.to(device=config.device), Q.to(dtype=dtype)).cpu(),
                    args[i],
                )

            mlp_ln_inputs, _ = get_signals(layer_adapter, args, kwargs)
            Q = svd_calc(mlp_ln_inputs, slicing_scheduler.get_attention_output_dimension(idx), ignore_masks)
            Q = Q.to(device=config.device, dtype=torch.float64)

            layer.attn_shortcut_Q = nn.Parameter(torch.matmul(layer.attn_shortcut_Q, Q.to(dtype=dtype)))

            project_attention_output(layer_adapter, Q, slicing_scheduler.get_attention_output_dimension(idx))

            layer.mlp_shortcut_Q = nn.Parameter(Q.T.clone().to(dtype=dtype))
            project_mlp_input(layer_adapter, Q, slicing_scheduler.get_mlp_input_dimension(idx))

            cleanup_memory()

            _, inps = get_signals(layer_adapter, args, kwargs)
            Q = svd_calc(inps, slicing_scheduler.get_mlp_output_dimension(idx), ignore_masks)

            layer.mlp_shortcut_Q = nn.Parameter(torch.matmul(layer.mlp_shortcut_Q, Q.to(dtype=dtype)))

            project_mlp_output(layer_adapter, Q, slicing_scheduler.get_mlp_output_dimension(idx))
            layer.mlp_shortcut_Q = nn.Parameter(layer.mlp_shortcut_Q)

            layer.to('cpu')

            cleanup_memory()

        if slicing_scheduler.do_slice_head:
            project_head(model_adapter, Q, slicing_scheduler.get_head_dimension())

        model_adapter.slicing_conf = slicing_scheduler.slicing_conf.clone()
        logging.info("Rotate and slice layers done")


@torch.no_grad()
def rotate(model_adapter: ModelAdapter, dataloader: torch.utils.data.DataLoader[torch.Tensor]) -> None:
    """
    Rotate a model.
    TODO: Make this gpu memory efficient.
    """
    model_adapter.model.eval()
    dtype = next(iter(model_adapter.model.parameters())).dtype  # Get the dtype of the model.

    # List of layers to rotate.
    layers = model_adapter.get_layers()

    # Get the input of the first layer norm and calculate the Q_1
    inps, args, kwargs = [], [], []
    for batch in dataloader:
        inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
        inps.append(inp_batch)
        args.append(args_batch)
        kwargs.append(kwargs_batch)

    _, Q_1 = pca_calc(inps)
    Q_1 = Q_1.to(device=config.device)

    # Rotate the embeddings.
    rotate_embeddings(model_adapter, Q_1)

    # Rotate the rest of the model.
    logging.info("Rotate layers")
    for layer_adapter in tqdm(layers, unit="layer", desc="Rotating"):
        layer = layer_adapter.layer
        # Extract the inputs and outputs of the second layernorm input and calculate the Q_3
        for i, inp in enumerate(inps):
            args[i] = layer_adapter.get_updated_args(inp, args[i])
        mlp_ln_inputs, outs = get_signals(layer_adapter, args, kwargs)
        _, Q_3 = pca_calc(mlp_ln_inputs)
        Q_3 = Q_3.to(device=config.device)
        _, Q_5 = pca_calc(outs)
        Q_5 = Q_5.to(device=config.device)

        # Rotate the Q, K and V matrices of the self-attention layer.
        rotate_attention_inputs(layer_adapter, Q_1)

        # Set the shortcut rotation matrix of the self-attention layer.
        layer.attn_shortcut_Q = nn.Parameter(torch.matmul(Q_1.clone().T, Q_3.clone()).to(device="cpu", dtype=dtype))

        # Rotate the Attention output matrix
        rotate_attention_output(layer_adapter, Q_3)

        # Rotate the MLP input
        rotate_mlp_input(layer_adapter, Q_3)

        # Set the shortcut rotation matrix of the MLP.
        layer.mlp_shortcut_Q = nn.Parameter(torch.matmul(Q_3.clone().T, Q_5.clone()).to(device="cpu", dtype=dtype))

        # Rotate MLP output
        rotate_mlp_output(layer_adapter, Q_5)

        # Run GC and cleanup GPU memory
        cleanup_memory()

        inps = outs  # The inputs to the next layer are the outputs from this one!
        Q_1 = Q_5  # first rotation in the next layer is the last one in this...

    rotate_head(model_adapter, Q_5)
    logging.info("Rotate layers done")


def slice_rotated_model(model_adapter: ModelAdapter, slicing_scheduler: SlicingScheduler | None = None) -> None:
    """
    TODO: Make this gpu memory efficient.
    """
    model_adapter.model.eval()
    layers = model_adapter.get_layers()
    if not slicing_scheduler:
        if model_adapter.slicing_conf.const_dimension is not None:
            # backward compatibility for when no config is available
            slicing_scheduler = ConstSlicingScheduler(model_adapter.slicing_conf.const_dimension)
            slicing_scheduler.setup(
                hidden_size=model_adapter.hidden_size,
                layers_num=len(layers),
                parallel_blocks=model_adapter.parallel_blocks,
            )
        else:
            slicing_scheduler = ConfigSlicingScheduler(model_adapter.slicing_conf)

    # slice embeddings
    slice_embeddings(model_adapter, slicing_scheduler.get_embedding_dimensions())

    # slice layers
    for i, layer_adapter in enumerate(layers):
        layer = layer_adapter.layer
        # slice attn weights 2nd dim, attn shortcut 1st dim
        slice_attention_inputs(layer_adapter, slicing_scheduler.get_attention_input_dimension(i))

        # slice mlp input 2nd dimension
        slice_mlp_input(layer_adapter, slicing_scheduler.get_mlp_input_dimension(i))

        # slice mlp shortcut 1st dimension
        # slice mlp shortcut
        if not model_adapter.parallel_blocks:
            layer.mlp_shortcut_Q = nn.Parameter(layer.mlp_shortcut_Q[: slicing_scheduler.get_mlp_input_dimension(i), :])

        # slice mlp weights 1st dimension
        slice_mlp_output(layer_adapter, slicing_scheduler.get_mlp_output_dimension(i))

        if model_adapter.parallel_blocks:  # parallel case
            layer.attn_shortcut_Q = nn.Parameter(
                layer.attn_shortcut_Q[:, : slicing_scheduler.get_attention_output_dimension(i, match_head_dim=True)]
            )
            slice_attention_output(
                layer_adapter, slicing_scheduler.get_attention_output_dimension(i, match_head_dim=True)
            )
        else:  # sequential case
            layer.attn_shortcut_Q = nn.Parameter(
                layer.attn_shortcut_Q[:, : slicing_scheduler.get_attention_output_dimension(i, match_head_dim=False)]
            ) 
            layer.mlp_shortcut_Q = nn.Parameter(
                layer.mlp_shortcut_Q[:, : slicing_scheduler.get_mlp_output_dimension(i)]
            )

            # slice attention weights 1st dimension
            slice_attention_output(
                layer_adapter, slicing_scheduler.get_attention_output_dimension(i, match_head_dim=False)
            )

    if slicing_scheduler.do_slice_head:
        slice_head(model_adapter, slicing_scheduler.get_head_dimension())


def random_orthogonal_upper_left(total_dim, upper_block_dim):
    """
    Create a square matrix where the upper left block is a random orthogonal matrix, and the remainder is the identity.
    """
    A = np.random.rand(upper_block_dim, upper_block_dim)
    Q, _ = np.linalg.qr(A)
    R = np.eye(total_dim)
    R[:upper_block_dim, :upper_block_dim] = Q
    return torch.from_numpy(R)


@torch.no_grad()
def pca_calc(
    X: list[torch.Tensor], ignore_masks: list[torch.Tensor] | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run PCA on a list of batched data. Returns the eigenvalues and eigenvectors.
    """
    # Run GC and cleanup GPU memory
    cleanup_memory()

    H = None
    for idx, X_batch in enumerate(X):
        if ignore_masks:
            X_batch[ignore_masks[idx] == 0] = 0

        X_batch = X_batch.double().to(device=config.device)
        H_batch = torch.sum(X_batch.mT @ X_batch, dim=0)  # sum over the batch dimension.
        #Note here: @ must perform a batched matrix multiply, though the documentation is unclear. This is 
        #indicated by the .mT attribute of the tensor, which is a batched matrix transpose.
        H = H_batch if H is None else H + H_batch

    damp = 0.01 * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[-1]).to(device=config.device)
    H[diag, diag] = H[diag, diag] + damp
    X_eig = torch.linalg.eigh(H)
    del H
    index = torch.argsort(X_eig[0], descending=True)
    eig_val = X_eig[0][index]
    eigen_vec = X_eig[1][:, index]
    return eig_val, eigen_vec

def unbatch(X: list[torch.Tensor]) -> torch.Tensor:
    """
    Concatenate a list of tensors along the batch dimension.
    """
    #Note: it's better to unbatch first to avoid multiple copy calls 
    unlisted = torch.vstack(X)
    #Note: flatten "folds" to the right and counts up numerically . 
    # flatten: ijk -> (i,j,k) flatten 2nd dimension: ijk -> i(j,k) flatten 1st dimension: ijk -> (i,j)k 
    unbatched = torch.flatten(unlisted, start_dim=0, end_dim=1)
    return unbatched

def svd_calc(X: list[torch.Tensor], rank: int, ignore_masks: list[torch.Tensor] | None = None) -> torch.Tensor: 
    """

    Calculate the SVD without streaming, using torch SVD algo
    """
    for idx, X_batch in enumerate(X):
        if ignore_masks:
            X_batch[ignore_masks[idx] == 0] = 0

    unbatched = unbatch(X)
    unbatched = unbatched.to(device=config.device, dtype=torch.float64)
    _,_,Vh = torch.linalg.svd(unbatched, full_matrices=False)
    Q = Vh.T
    return Q[:, :rank]
