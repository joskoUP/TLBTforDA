% This is a script to create the prior compatibility comparison example
% (Figure 1) of our paper. The code for balancing Bayesian inference is
% taken from https://github.com/elizqian/balancing-bayesian-inference.

clear; close all
addpath('models')
addpath('external_functions')

%% define LTI model
% heat equation
load('heat-cont.mat');
d           = size(A,1);
sig_obs     = 0.008;
d_out       = size(C,1);

%% compute prior = Reachability Gramian
main_diag   = ones(1,d);
first_diag  = 0.5*randn(1,d-1);
second_diag = 0.25*randn(1,d-2);
L_pr        = diag(main_diag) + diag(first_diag,1) + diag(second_diag,2);
% non-compatible Prior
Gamma_pr    = L_pr*L_pr';
M           = A*Gamma_pr+Gamma_pr*A';
if sum(real(eig(M))>eps) > 0
    disp('The prior covariance L_pr is not prior compatible.')
end

% draw random initial condition from non-compatible prior
x0          = L_pr*randn(d,1);
% generate random data from multiple initial conditions
num_reps    = 100;
x0_all      = L_pr*randn(d,num_reps);

%% make prior prior-compatible by technique from Qian et.al.sect. 4.1.2
% choose truncation tolerance when real parts are considered as zero, is
% necessary due to small numerical errors induced
tol = 1e-8;
L_pr_comp   = minmodPriorCompat(Gamma_pr,A)';
Gamma_pr_comp    = L_pr_comp*L_pr_comp';
M_comp      = A*Gamma_pr_comp+Gamma_pr_comp*A';
if sum(real(eig(M_comp))>tol) == 0
    disp('The modified covariance L_pr_comp is prior-compatible.')
end

% draw random initial condition from compatible prior
x0_comp          = L_pr_comp*randn(d,1);
% generate random data from multiple initial conditions
x0_all_comp      = L_pr_comp*randn(d,num_reps);

%% compute infinite Obs Gramian
% helper matrix    
F           = C./sig_obs;
L_Q         = lyapchol(A',F')';
Q_inf       = L_Q*L_Q';

%% define time frame for inference problem
% measurement times and noise scaling
T           = 10;
T_length    = length(T);
dt_obs      = 5e-3;       
% making dt_obs bigger makes Spantini eigvals decay faster
n           = round(T/dt_obs);
% relative noise scaling
scl_sig_obs = 0.1;   

%% define parameters for rational Krylov
M = speye(d);
% handling of M-matrix
opts.ML     = [];   opts.MU = []; % precomputed factors of M 
opts.facM   = 1;    % use pivoted sparse LU/Cholesky, 1- sparse LU/Choles
timeCholM   = 0;
tol         = 1e-8; % stopping tolerance
maxit       = 100;

figure; clf  
for t = 1:T_length
    if ~exist('sig_obs','var')
        sig_obs = scl_sig_obs*max(abs(reshape(y,d_out,n(t))),[],2);
    end
    sig_obs_long = repmat(sig_obs,n(t),1);

    %% define full forward model 
    G       = zeros(n(t)*d_out,d);
    iter    = expm(A*dt_obs);
    temp    = C;
    for i   = 1:n(t)
        temp                        = temp*iter;
        G((i-1)*d_out+1:i*d_out,:)  = temp;
    end

    %% compute Fisher info
    Go  = G./sig_obs_long;
    H   = Go'*Go;

    %% generate measurements for compatible prior
    % single measurement
    y       = G*x0;
    m       = y + sig_obs_long.*randn(n(t)*d_out,1);
    % multiple measurements for Bayes risk
    y_all   = G*x0_all;
    m_all   = y_all + sig_obs_long.*randn(n(t)*d_out,num_reps);

    %% generate measurements for compatible prior
    % single measurement
    y_comp       = G*x0_comp;
    m_comp       = y_comp + sig_obs_long.*randn(n(t)*d_out,1);
    % multiple measurements for Bayes risk
    y_all_comp   = G*x0_all_comp;
    m_all_comp   = y_all_comp + sig_obs_long.*randn(n(t)*d_out,num_reps);

    %% compute true posterior (for true, non-compatible prior)
    full_rhs        = G'*(m./(sig_obs_long.^2));
    full_rhs_all    = G'*(m_all./(sig_obs_long.^2));

    L_prinv         = inv(L_pr); 
 
    R_posinv        = qr([Go; L_prinv],0);
    R_posinv        = triu(R_posinv(1:d,:)); % Pull out upper triangular factor
    R_pos_true      = inv(R_posinv);
    mupos_true      = R_posinv\(R_posinv'\full_rhs);
    mupos_true_all  = R_posinv\(R_posinv'\full_rhs_all);

    %% compute posterior (for modified compatible prior)
    full_rhs_comp        = G'*(m_comp./(sig_obs_long.^2));
    full_rhs_all_comp    = G'*(m_all_comp./(sig_obs_long.^2));

    L_prinv_comp         = inv(L_pr_comp); 
 
    R_posinv_comp        = qr([Go; L_prinv_comp],0);
    R_posinv_comp        = triu(R_posinv_comp(1:d,:)); % Pull out upper triangular factor
    R_pos_true_comp      = inv(R_posinv_comp);
    mupos_true_comp      = R_posinv_comp\(R_posinv_comp'\full_rhs_comp);
    mupos_true_all_comp  = R_posinv_comp\(R_posinv_comp'\full_rhs_all_comp);

    %% compute posterior approximations and errors
    r_vals      = 1:20;
    rmax        = max(r_vals);

    % (H,Gamma_pr^-1) computations non-compatible prior
    [~,R]       = qr(Go,0);         % compute a square root factorization of H
    LG          = R';
    [V,S,W]     = svd(LG'*L_pr,0);  
    tau         = diag(S);
    What        = L_pr*W;           % spantini directions
    Wtilde      = L_pr'\W;
    S           = S(1:rmax,1:rmax);   
    delH        = diag(S);
    Siginvsqrt  = diag(1./sqrt(delH));
    SrH         = (Siginvsqrt*V(:,1:rmax)'*LG')';
    TrH         = L_pr*W(:,1:rmax)*Siginvsqrt; % balancing transformation
    A_BTH       = SrH'*A*TrH;
    C_BTH       = C*TrH;

    % (H,Gamma_pr^-1) computations compatible prior
    [V_comp,S_comp,W_comp]  = svd(LG'*L_pr_comp,0);
    tau_comp                = diag(S_comp);
    What_comp               = L_pr_comp*W_comp;    % spantini directions
    Wtilde_comp             = L_pr_comp'\W_comp;
    S_comp                  = S_comp(1:rmax,1:rmax);   
    delH_comp               = diag(S_comp);
    Siginvsqrt_comp         = diag(1./sqrt(delH_comp));
    SrH_comp                = (Siginvsqrt_comp*V_comp(:,1:rmax)'*LG')';
    TrH_comp                = L_pr_comp*W_comp(:,1:rmax)*Siginvsqrt_comp; % balancing transformation
    A_BTH_comp              = SrH_comp'*A*TrH_comp;
    C_BTH_comp              = C*TrH_comp;
    
    %% balancing with Q_infty
    [V,S,W]     = svd(L_Q'*L_pr); 
    S           = S(1:rmax,1:rmax);
    delQ        = diag(S);
    Siginvsqrt  = diag(1./sqrt(delQ));
    Sr          = (Siginvsqrt*V(:,1:rmax)'*L_Q')';
    Tr          = L_pr*W(:,1:rmax)*Siginvsqrt; % balancing transformation
    A_BTQ       = Sr'*A*Tr;
    C_BTQ       = C*Tr;

    %% balancing with Q_infty and modified compatible prior
    [V_comp,S_comp,W_comp]  = svd(L_Q'*L_pr_comp); 
    S_comp                  = S_comp(1:rmax,1:rmax);
    delQ_comp               = diag(S_comp);
    Siginvsqrt_comp         = diag(1./sqrt(delQ_comp));
    Sr_comp                 = (Siginvsqrt_comp*V_comp(:,1:rmax)'*L_Q')';
    Tr_comp                 = L_pr_comp*W_comp(:,1:rmax)*Siginvsqrt_comp; % balancing transformation
    A_BTQ_comp              = Sr_comp'*A*Tr_comp;
    C_BTQ_comp              = C*Tr_comp;

    %% compute posterior approximations
    f_dist                                      = zeros(length(r_vals),5);
    % [mu_LR, mu_LR_comp, mu_BTQ, mu_BTQ_comp]    = deal(zeros(d,length(r_vals)));
    mu_errs                                     = zeros(length(r_vals),5);
    for rr = 1:length(r_vals)
        r           = r_vals(rr);
        
        %% Spantini posterior quantities (non-compatible prior)
        % Spantini approx posterior covariance 
        Rpos_sp         = What*diag(sqrt([1./(1+tau(1:r).^2); ones(d-r,1)]));
        Gpos_sp         = What*diag([1./(1+tau(1:r).^2); ones(d-r,1)])*What';

        % Bayes risk and Foerstner error derived from non-compatible prior
        f_dist(rr,1)    = forstner(Rpos_sp,R_pos_true,'sqrt');
        % Spantini approx posterior means
        Pi_r            = What(:,1:r)*Wtilde(:,1:r)';
        temp_sp_1       = Gpos_sp*Pi_r'*full_rhs_all;
        temp_sp_1       = R_pos_true\(temp_sp_1 - mupos_true_all);
        mu_errs(rr,1)   = mean(sqrt(sum(temp_sp_1.^2)));
    
        %% Q_infty posterior quantities (non-compatible prior)
        % Balancing with Q_infty - generate G_BTQ,H_BTQ
        G_BTQ           = zeros(n(t)*d_out,r);
        iter            = expm(A_BTQ(1:r,1:r)*dt_obs);
        temp            = C_BTQ(:,1:r);
        for i = 1:n(t)
            temp                            = temp*iter;
            G_BTQ((i-1)*d_out+1:i*d_out,:)  = temp;
        end
        G_BTQ           = G_BTQ*Sr(:,1:r)';
        G_BTQo          = G_BTQ./sig_obs_long;
        H_BTQ           = G_BTQo'*G_BTQo;

        % Balancing with Q_infty - compute posterior covariance and mean
        R_posinv        = qr([G_BTQo; L_prinv],0);
        R_posinv        = triu(R_posinv(1:d,:)); % Pull out upper triangular factor
        R_pos_BTQ       = inv(R_posinv);
        Gpos_BTQ        = R_pos_BTQ*R_pos_BTQ';
    
        % Bayes risk and Foerstner error derived from non-compatible prior
        f_dist(rr,2)    = forstner(R_pos_BTQ,R_pos_true,'sqrt');
        temp_BTQ        = Gpos_BTQ*G_BTQ'*(m_all./(sig_obs_long.^2));
        temp_BTQ        = R_pos_true\(temp_BTQ - mupos_true_all);
        mu_errs(rr,2)   = mean(sqrt(sum(temp_BTQ.^2)));

        %%  Q_infty posterior quantities (modified compatible prior)
        % Balancing with Q_infty - generate G_BTQ_comp,H_BTQ_comp
        G_BTQ_comp        = zeros(n(t)*d_out,r);
        iter              = expm(A_BTQ_comp(1:r,1:r)*dt_obs);
        temp              = C_BTQ_comp(:,1:r);
        for i = 1:n(t)
            temp                                = temp*iter;
            G_BTQ_comp((i-1)*d_out+1:i*d_out,:) = temp;
        end
        G_BTQ_comp        = G_BTQ_comp*Sr_comp(:,1:r)';
        G_BTQo_comp       = G_BTQ_comp./sig_obs_long;
        H_BTQ_comp        = G_BTQo_comp'*G_BTQo_comp;

        % Balancing with Q_infty - compute posterior covariance and mean
        R_posinv_comp   = qr([G_BTQo_comp; L_prinv_comp],0);
        R_posinv_comp   = triu(R_posinv_comp(1:d,:)); % Pull out upper triangular factor
        R_pos_BTQ_comp  = inv(R_posinv_comp);
        Gpos_BTQ_comp   = R_pos_BTQ_comp*R_pos_BTQ_comp';

        % Bayes risk and Foerstner error derived from non-compatible prior
        f_dist(rr,3)    = forstner(R_pos_BTQ_comp,R_pos_true,'sqrt');
        temp_BTQ_comp   = Gpos_BTQ_comp*G_BTQ_comp'*(m_all./(sig_obs_long.^2));
        temp_BTQ_comp   = R_pos_true\(temp_BTQ_comp - mupos_true_all);
        mu_errs(rr,3)   = mean(sqrt(sum(temp_BTQ_comp.^2)));

        % Bayes risk and Foerstner error derived from compatible prior
        f_dist(rr,4)    = forstner(R_pos_BTQ_comp,R_pos_true_comp,'sqrt');
        temp_BTQ_comp2  = Gpos_BTQ_comp*G_BTQ_comp'*(m_all_comp./(sig_obs_long.^2));
        temp_BTQ_comp2  = R_pos_true_comp\(temp_BTQ_comp2 - mupos_true_all_comp);
        mu_errs(rr,4)   = mean(sqrt(sum(temp_BTQ_comp2.^2)));

        %% Spantini posterior quantities (modified compatible prior)
        % Spantini approx posterior covariance 
        Rpos_sp_comp    = What_comp*diag(sqrt([1./(1+tau_comp(1:r).^2); ones(d-r,1)]));
        Gpos_sp_comp    = What_comp*diag([1./(1+tau_comp(1:r).^2); ones(d-r,1)])*What_comp';

        % Bayes risk and Foerstner error derived from compatible prior
        f_dist(rr,5)    = forstner(Rpos_sp_comp,R_pos_true_comp,'sqrt');
        % Spantini approx posterior means
        Pi_r_comp       = What_comp(:,1:r)*Wtilde_comp(:,1:r)';
        temp_sp_2       = Gpos_sp_comp*Pi_r_comp'*full_rhs_all_comp;
        temp_sp_2       = R_pos_true_comp\(temp_sp_2 - mupos_true_all_comp);
        mu_errs(rr,5)   = mean(sqrt(sum(temp_sp_2.^2)));
    end

    %% plots
    % Warning if complex parts of Förstner distances are nontrivial
    if ~isempty(find(abs(imag(f_dist))>eps*abs(real(f_dist)), 1))
        warning('Significant imaginary parts found in Forstner distance')
    end
    % Otherwise imaginary parts are trivial artifacts of generalized eig
    f_dist  = real(f_dist);
 
    %% plot posterior covariance Forstner errors
    subplot(T_length,2,2*t)
    semilogy(r_vals,f_dist(:,3),'o','Color','magenta'); hold on
    semilogy(r_vals,f_dist(:,4),'+','Color','magenta');
    semilogy(r_vals,f_dist(:,2),'o','Color',[0.8500    0.3250    0.0980]);
    semilogy(r_vals,f_dist(:,5),'+','Color','cyan');
    semilogy(r_vals,f_dist(:,1),'o','Color','blue'); 
%     
%     if you add this for BT-H, add 'BT-H' to the legend
    grid on
    if t == 1
        title('$\Gamma_{pos}$ F\"orstner error','interpreter','latex','fontsize',20)    
    end
    if t == T_length
        xlabel('$r$','interpreter','latex','fontsize',13)
        legend({'BT-Q C-NC','BT-Q C-C','BT-Q NC-NC','OLR C-C','OLR NC-NC'},'interpreter','latex','fontsize',13,'Location','southwest')
        legend boxoff
    end
    ylim([1e-15 1000])
    set(gca,'fontsize',13,'ticklabelinterpreter','latex')

    %% plot posterior mean errors in Gpos^-1 norm
    posnormref = mean(sqrt(sum((R_pos_true\mupos_true_all).^2)));
    subplot(T_length,2,2*t-1)
    semilogy(r_vals,mu_errs(:,3)/posnormref,'o','Color','magenta'); hold on
    semilogy(r_vals,mu_errs(:,4)/posnormref,'+','Color','magenta');
    semilogy(r_vals,mu_errs(:,2)/posnormref,'o','Color',[0.8500    0.3250    0.0980]);
    semilogy(r_vals,mu_errs(:,5)/posnormref,'+','Color','cyan');
    semilogy(r_vals,mu_errs(:,1)/posnormref,'o','Color','blue'); 
%     if you add this for BT-H, add 'BT-H' to the legend
    grid on
   if t == 1 
       title('$\mu_{pos}$  Bayes risk','interpreter','latex','fontsize',20)
    end
    if t == T_length
         xlabel('$r$','interpreter','latex','fontsize',13)
        legend({'BT-Q C-NC','BT-Q C-C','BT-Q NC-NC','OLR C-C','OLR NC-NC'},'interpreter','latex','fontsize',13,'Location','southwest')
        legend boxoff
    end
    ylim([1e-8 1])
    set(gca,'fontsize',13,'ticklabelinterpreter','latex')

end