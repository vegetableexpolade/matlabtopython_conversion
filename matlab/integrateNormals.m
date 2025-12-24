function depth = integrateNormals(normals, method)
% INTEGRATENORMALS Integrate surface normals to recover depth
%   DEPTH = INTEGRATENORMALS(NORMALS, METHOD) integrates surface normals
%   to estimate the depth map (height field).
%
%   Inputs:
%       NORMALS - H x W x 3 array of surface normals
%       METHOD - Integration method: 'average' or 'poisson' (default: 'average')
%
%   Output:
%       DEPTH - H x W depth map
%
%   The function uses gradient integration to reconstruct the surface.

    if nargin < 2
        method = 'average';
    end
    
    [height, width, ~] = size(normals);
    
    % Extract normal components
    nx = normals(:, :, 1);
    ny = normals(:, :, 2);
    nz = normals(:, :, 3);
    
    % Avoid division by zero
    nz(abs(nz) < 1e-6) = 1e-6;
    
    % Compute gradients: p = -nx/nz, q = -ny/nz
    p = -nx ./ nz;
    q = -ny ./ nz;
    
    if strcmp(method, 'average')
        % Simple path integration using averaging
        depth = zeros(height, width);
        
        % Integrate from top-left corner
        % Along first row
        for j = 2:width
            depth(1, j) = depth(1, j-1) + p(1, j-1);
        end
        
        % Along first column
        for i = 2:height
            depth(i, 1) = depth(i-1, 1) + q(i-1, 1);
        end
        
        % Fill in the rest using average of two paths
        for i = 2:height
            for j = 2:width
                from_left = depth(i, j-1) + p(i, j-1);
                from_top = depth(i-1, j) + q(i-1, j);
                depth(i, j) = (from_left + from_top) / 2;
            end
        end
        
    elseif strcmp(method, 'poisson')
        % Poisson integration using Laplacian
        % This is a simplified version
        % Compute divergence of gradient field
        [px, ~] = gradient(p);
        [~, qy] = gradient(q);
        div = px + qy;
        
        % Solve Poisson equation using DCT (discrete cosine transform)
        D = dct2(div);
        
        [x, y] = meshgrid(1:width, 1:height);
        denom = 2 * (cos(pi * (x-1) / width) + cos(pi * (y-1) / height) - 2);
        denom(1, 1) = 1;  % Avoid division by zero at DC component
        
        Z = D ./ denom;
        Z(1, 1) = 0;  % Set DC component to zero (mean depth = 0)
        
        depth = idct2(Z);
    else
        error('Unknown integration method: %s', method);
    end
end
