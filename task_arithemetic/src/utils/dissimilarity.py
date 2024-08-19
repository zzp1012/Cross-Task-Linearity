import torch

class DissimilarityMetric:
    """compute the distance of two featuremaps
    """
    def __init__(self, metric):
        self.__metric = metric

    def __call__(self, A, B, **kwargs):
        if self.__metric == "vanilla":
            return self.__vanilla(A, B)
        elif self.__metric == "cosine":
            return self.__cosine_similarity(A, B, **kwargs)
        elif self.__metric == "abs_div":
            return self.__abs_div(A, B, **kwargs)

        A = self.__preprocess(A)
        B = self.__preprocess(B)
        if self.__metric == "lin_cka":
            return self.__lin_cka_dist(A, B)
        elif self.__metric == "lin_cka_prime":
            return self.__lin_cka_prime_dist(A, B)
        elif self.__metric == "procrustes":
            return self.__procrustes(A, B)
        else:
            raise ValueError("Unknown metric")

    def __preprocess(self, X: torch.Tensor) -> torch.Tensor:
        """preprocess the featuremap
            1. flatten the featuremap
            2. transpose the featuremap
            3. center the featuremap
            4. normalize the featuremap 
        
        Args:
            X (torch.Tensor): the featuremap, shape: (N, ...)

        Return:
            X (torch.Tensor): the preprocessed featuremap, shape: (N, D)
        """
        # flatten the featuremap
        X = X.view(X.shape[0], -1) # shape: (N, D)
        # transpose the featuremap
        X = X.t() # shape: (D, N)
        # centering
        X = X - X.mean(dim=-1, keepdim=True) # shape: (D, N)
        # normalize with the Frobenius norm
        X = X / torch.norm(X, p="fro") # shape: (D, N)
        return X

    def __lin_cka_dist(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """compute the linear CKA distance between two featuremaps
        Args:
            A (torch.Tensor): the featuremap A, shape: (D, N)
            B (torch.Tensor): the featuremap B, shape: (D', N)

        Return:
            dist (torch.Tensor): the distance between A and B
        """
        similarity = torch.norm(B @ A.t(), p="fro") ** 2
        normalization = torch.norm(A @ A.t(), p="fro") * torch.norm(B @ B.t(), p="fro")
        return 1 - similarity / normalization

    def __lin_cka_prime_dist(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Computes Linear CKA prime distance bewteen representations A and B
        The version here is suited to D, D' >> N

        Args:
            A (torch.Tensor): the featuremap A, shape: (D, N)
            B (torch.Tensor): the featuremap B, shape: (D', N)

        Return:
            dist (torch.Tensor): the distance between A and B
        """
        if A.shape[0] > A.shape[1]: # D > N
            At_A = A.t() @ A # shape: (N, N) O(n * n * a)
            Bt_B = B.t() @ B # shape: (N, N) O(n * n * a)
            numerator = torch.sum((At_A - Bt_B) ** 2)
            denominator = torch.sum(A ** 2) ** 2 + torch.sum(B ** 2) ** 2
            return numerator / denominator
        else:
            similarity = torch.norm(B @ A.t(), p="fro") ** 2
            denominator = torch.sum(A ** 2) ** 2 + torch.sum(B ** 2) ** 2
            return 1 - 2 * similarity / denominator

    def __procrustes(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Compute the Procrustes distance between two featuremaps
        Args:
            A (torch.Tensor): the featuremap A, shape: (D, N)
            B (torch.Tensor): the featuremap B, shape: (D', N)

        Return:
            dist (torch.Tensor): the distance between A and B
        """
        A_sq_frob = torch.sum(A ** 2)
        B_sq_frob = torch.sum(B ** 2)
        nuc = torch.norm(A @ B.t(), p="nuc")  # O(p * p * n)
        return A_sq_frob + B_sq_frob - 2 * nuc

    def __vanilla(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """compute the vanilla distance between two featuremaps, Frobenius norm

        Args:
            A (torch.Tensor): the featuremap A, shape: (N, D)
            B (torch.Tensor): the featuremap B, shape: (N, D)
        """
        assert A.shape == B.shape, \
            f"the shape of A - {A.shape} should be the same as the shape of B - {B.shape}"
        A = A.view(A.shape[0], -1) # shape: (N, D)
        B = B.view(B.shape[0], -1) # shape: (N, D)

        def norm_square(A: torch.Tensor) -> torch.Tensor:
            return torch.sum(A ** 2) # shape: (1, )

        return norm_square(A - B) / torch.norm(A, p="fro") / torch.norm(B, p="fro")

    def __abs_div(self, A: torch.Tensor, B: torch.Tensor, **kwargs) -> torch.Tensor:
        get_coef = kwargs.get("get_coef", False)
        
        assert A.shape == B.shape, \
            f"the shape of A - {A.shape} should be the same as the shape of B - {B.shape}"
        A = A.view(A.shape[0], -1).double() # shape: (N, D)
        B = B.view(B.shape[0], -1).double() # shape: (N, D)
        
        dist = torch.sum(torch.abs(A)) / torch.sum(torch.abs(B))
        
        coef = 0.
        
        if get_coef:
            return dist, coef
        else:
            return dist   
        
    def __cosine_similarity(self, A: torch.Tensor, B: torch.Tensor, **kwargs) -> torch.Tensor:
        """compute the cosine similarity between two matrices

        dist = 1 - <A, B> / (||A|| * ||B||)

        Args:
            A (torch.Tensor): the featuremap A, shape: (N, D)
            B (torch.Tensor): the featuremap B, shape: (N, D)
            kwargs: the keyword arguments

        Return:
            dist (torch.Tensor): the distance between A and B
        """
        get_coef = kwargs.get("get_coef", False)

        assert A.shape == B.shape, \
            f"the shape of A - {A.shape} should be the same as the shape of B - {B.shape}"

        A = A.view(A.shape[0], -1).double() # shape: (N, D)
        B = B.view(B.shape[0], -1).double() # shape: (N, D)

        # compute the frobenius inner product of A and B
        inner_product = torch.sum(A * B) # shape: (1, 1)
        # compute the frobenius norm of A and B
        A_norm = torch.norm(A, p="fro") # shape: (1, 1)
        B_norm = torch.norm(B, p="fro") # shape: (1, 1)

        # cal the distance
        dist = 1 - torch.abs(inner_product) / (A_norm * B_norm)

        # compute the coefficient
        coef = inner_product / (B_norm ** 2)

        assert torch.abs(inner_product) <= A_norm * B_norm * (1 + 1e-10), \
            f"the inner product - {inner_product} should be less than the product of the norm - {A_norm * B_norm}"

        if get_coef:
            return dist, coef
        else:
            return dist


class DissimilarityMetricOverSamples:
    """compute the distance of two featuremaps, for each sample
    """
    def __init__(self, metric):
        self.__metric = metric

    def __call__(self, A, B, **kwargs):
        if self.__metric == "vanilla":
            return self.__vanilla(A, B)
        elif self.__metric == "cosine":
            return self.__cosine_similarity(A, B, **kwargs)
        elif self.__metric == "abs_div":
            return self.__abs_div(A, B, **kwargs)

    def __vanilla(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """compute the vanilla distance between two featuremaps, Frobenius norm

        Args:
            A (torch.Tensor): the featuremap A, shape: (N, D)
            B (torch.Tensor): the featuremap B, shape: (N, D)
        """
        assert A.shape == B.shape, \
            f"the shape of A - {A.shape} should be the same as the shape of B - {B.shape}"
        A = A.view(A.shape[0], -1) # shape: (N, D)
        B = B.view(B.shape[0], -1) # shape: (N, D)

        def norm_square(A: torch.Tensor) -> torch.Tensor:
            return torch.sum(A ** 2, dim=-1) # shape: (N, )
        return norm_square(A - B) / torch.norm(A, p="fro", dim=-1) / torch.norm(B, p="fro", dim=-1) # shape: (N, )

    def __abs_div(self, A: torch.Tensor, B: torch.Tensor, **kwargs) -> torch.Tensor:
        get_coef = kwargs.get("get_coef", False)
        
        assert A.shape == B.shape, \
            f"the shape of A - {A.shape} should be the same as the shape of B - {B.shape}"
        A = A.view(A.shape[0], -1).double() # shape: (N, D)
        B = B.view(B.shape[0], -1).double() # shape: (N, D)
        
        dist = torch.sum(torch.abs(A),dim=-1) / torch.sum(torch.abs(B),dim=-1)
        
        coef = 0.
        
        if get_coef:
            return dist, coef
        else:
            return dist

    def __cosine_similarity(self, A: torch.Tensor, B: torch.Tensor, **kwargs) -> torch.Tensor:
        """compute the cosine similarity between two matrices

        dist = 1 - <A, B> / (||A|| * ||B||)

        Args:
            A (torch.Tensor): the featuremap A, shape: (N, D)
            B (torch.Tensor): the featuremap B, shape: (N, D)
            kwargs: the keyword arguments

        Return:
            dist (torch.Tensor): the distance between A and B
        """
        get_coef = kwargs.get("get_coef", False)

        assert A.shape == B.shape, \
            f"the shape of A - {A.shape} should be the same as the shape of B - {B.shape}"

        A = A.view(A.shape[0], -1).double() # shape: (N, D)
        B = B.view(B.shape[0], -1).double() # shape: (N, D)

        # compute the frobenius inner product of A and B
        inner_product = torch.sum(A * B, dim=-1) # shape: (N, )
        # compute the frobenius norm of A and B
        A_norm = torch.norm(A, p="fro", dim=-1) # shape: (N, )
        B_norm = torch.norm(B, p="fro", dim=-1) # shape: (N, )

        # cal the distance
        dist = 1 - torch.abs(inner_product) / (A_norm * B_norm) # shape: (N, )

        # compute the coefficient
        coef = inner_product / (B_norm ** 2) # shape: (N, )

        if get_coef:
            return dist, coef
        else:
            return dist


if __name__ == "__main__":
    # set the seed
    torch.manual_seed(0)
    A = torch.randn(2, 3, 4, 5)
    B = torch.randn(2, 3, 4, 5)
    
    metric = DissimilarityMetric("cosine")
    print(metric(A, B))