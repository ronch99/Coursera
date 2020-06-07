function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

m = size(X, 1);
K = size(centroids, 1);
idx = ones(m, 1);

for i = 1:m
    x = X(i, :);
    d = sum((x - centroids(1, :)).^2);
    for j = 2:K
        d_ = sum((x - centroids(j, :)).^2);
        if d_ < d
            idx(i) = j;
            d = d_;
        end
    end
end

end

