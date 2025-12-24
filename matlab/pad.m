function padded = pad(img, padsize, method)
% PAD Pad an array with specified values
%   PADDED = PAD(IMG, PADSIZE, METHOD) pads the input array IMG with
%   PADSIZE elements on each side using the specified METHOD.
%
%   Inputs:
%       IMG - Input array (can be 2D or 3D)
%       PADSIZE - Size of padding on each side (scalar or [rows, cols])
%       METHOD - Padding method: 'constant', 'replicate', or 'symmetric'
%
%   Output:
%       PADDED - Padded array
%
%   Example:
%       A = [1 2; 3 4];
%       B = pad(A, 1, 'constant');

    % Default method is 'constant' with value 0
    if nargin < 3
        method = 'constant';
    end
    
    % Convert scalar padsize to [rows, cols]
    if isscalar(padsize)
        padsize = [padsize, padsize];
    end
    
    % Get dimensions
    [rows, cols, channels] = size(img);
    
    % Calculate output size
    new_rows = rows + 2 * padsize(1);
    new_cols = cols + 2 * padsize(2);
    
    % Initialize output array
    if strcmp(method, 'constant')
        padded = zeros(new_rows, new_cols, channels, 'like', img);
        % Copy original image to center
        padded(padsize(1)+1:padsize(1)+rows, padsize(2)+1:padsize(2)+cols, :) = img;
        
    elseif strcmp(method, 'replicate')
        padded = zeros(new_rows, new_cols, channels, 'like', img);
        % Copy original image to center
        padded(padsize(1)+1:padsize(1)+rows, padsize(2)+1:padsize(2)+cols, :) = img;
        
        % Replicate edges
        % Top and bottom
        for i = 1:padsize(1)
            padded(i, padsize(2)+1:padsize(2)+cols, :) = img(1, :, :);
            padded(new_rows-i+1, padsize(2)+1:padsize(2)+cols, :) = img(end, :, :);
        end
        
        % Left and right
        for j = 1:padsize(2)
            padded(:, j, :) = padded(:, padsize(2)+1, :);
            padded(:, new_cols-j+1, :) = padded(:, padsize(2)+cols, :);
        end
        
    elseif strcmp(method, 'symmetric')
        padded = zeros(new_rows, new_cols, channels, 'like', img);
        % Copy original image to center
        padded(padsize(1)+1:padsize(1)+rows, padsize(2)+1:padsize(2)+cols, :) = img;
        
        % Mirror edges
        % Top
        if padsize(1) > 0
            mirror_rows = min(padsize(1), rows);
            padded(1:mirror_rows, padsize(2)+1:padsize(2)+cols, :) = ...
                flipud(img(1:mirror_rows, :, :));
        end
        
        % Bottom
        if padsize(1) > 0
            mirror_rows = min(padsize(1), rows);
            padded(end-mirror_rows+1:end, padsize(2)+1:padsize(2)+cols, :) = ...
                flipud(img(end-mirror_rows+1:end, :, :));
        end
        
        % Left
        if padsize(2) > 0
            mirror_cols = min(padsize(2), cols);
            padded(:, 1:mirror_cols, :) = ...
                fliplr(padded(:, padsize(2)+1:padsize(2)+mirror_cols, :));
        end
        
        % Right
        if padsize(2) > 0
            mirror_cols = min(padsize(2), cols);
            padded(:, end-mirror_cols+1:end, :) = ...
                fliplr(padded(:, padsize(2)+cols-mirror_cols+1:padsize(2)+cols, :));
        end
    else
        error('Unknown padding method: %s', method);
    end
end
