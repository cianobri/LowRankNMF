function [X] = BNMD_lowrank(S, D, thresh)
% low rank beta-NMD with beta = 0.5 and fixed dictionary
% stops on convergence criteria
% Input
% S - spectrogram
% D - dictionary
% thresh - value for singular value thresholding
% Output 
% X - coefficient matrix


max_iter = 4000;
err = 0.000001;
iter = 0;
converged = 0;

%% initialise

X = 0.01 * ones(size(D' * S));
R = D * X;
cost = get_cost(S, R);


%% Perform optimisation

while ~converged && iter < max_iter

    iter = iter + 1;

    RR = 1./(sqrt(R));
    RRR = RR .* (S./(R));
           
    X = X .* (D'*RRR) ./ (D'*RR);

    % low rank approximation:
    if any(isnan(X(:))) || any(isinf(X(:)))
        %disp('Nan or inf :(');
    else
        [u,s,v] = svd(X);
        st = soft_thresh(diag(s(:,1:size(s,1))),thresh); 
        X  = u*[diag(st) zeros(size(st,1), size(v,2)-size(diag(st),2))]*v'; 
    end
    %todo: update dictionary matrix D
    if mod(iter,0)==0
        R = D*X;
        Rs = sqrt(R); RR = Rs./R;
        A = (S.*RR)*X';
        B = R*X';
        D = D .* ( (A + 0.1) ./ (B + 0.1) ).^0.4;
    end
    
    R = D * X;
    
    %% check for convergence
    
    if mod(iter, 5) == 0
        new_cost = get_cost(S, R);
        if (cost-new_cost) / cost < err && iter > 10
           converged = 1;
           %disp('change in obj val. less than limit');
        end
        cost = new_cost;
    end
    
end

end
    


function cost = get_cost(Data, R)

A = sqrt(Data);
B = sqrt(R);
C = A-B;
C = C.^2;
cost = 2 * sum(sum(C ./ B));

end



