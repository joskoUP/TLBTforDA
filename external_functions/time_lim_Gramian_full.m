% function TIME_LIM_GRAMIAN_FULL.m
% Computes the full time-limited Gramian for matrices A, B and end time t.
% opts can be 'cont' for continuous TL_Gramians or 'disc' for the discrete
% setting. If nothing/none of the two is given, continuous setting is chosen. 

function Pt_lyap = time_lim_Gramian_full(A,B,t,opts)

if nargin < 4
    fprintf('No setting was given. Continuous TL-Gramians are computed.\n');
    opts = 'cont';
end

if strcmp (opts,'cont')==1
    BB = B*B';
    rhs = BB - expm(A*t) * BB * expm(A'*t);
    Pt_lyap =lyap(A,rhs);

elseif strcmp (opts,'disc')==1
    BB = B*B';
    rhs = BB - (A^t) * BB * (A'^t);
    Pt_lyap = dlyap(A,rhs);

else
    fprintf('No setting was given. Continuous TL-Gramians are computed.\n');
    BB = B*B';
    rhs = BB - expm(A*t) * BB * expm(A'*t);
    Pt_lyap =lyap(A,rhs);
end
end