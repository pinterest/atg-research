from __future__ import annotations

import torch


class TfLazyLinear(torch.nn.LazyLinear):
    """
    A torch.nn.Linear module with lazy initialization. We initialize the
    variables with similar distribution to what TF does for Dense.
    """

    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params() and self.in_features != 0:
            torch.nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                torch.nn.init.zeros_(self.bias)


def linear_tf(in_features, out_features, bias=True):
    """
    Initialize nn.Linear with similar distribution to what TF does for Dense
    """
    x = torch.nn.Linear(in_features, out_features, bias=bias)

    torch.nn.init.xavier_uniform_(x.weight)
    if bias:
        torch.nn.init.zeros_(x.bias)
    return x
