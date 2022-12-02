% function FORSTNER.m
% computes the distance between two matrices in the Förstner-metric
% taken from https://github.com/elizqian/balancing-bayesian-inference

function nm = forstner(A,B,form)
if strcmp(form, 'sqrt')
    sig = gev_sqrt(A,B);
elseif strcmp(form, 'spd')
    sig = eig(A,B,'chol');
end
    nm = sum(log(sig).^2);
end