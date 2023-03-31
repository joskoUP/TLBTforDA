% This is a script to create the heat equation example (Figure 2) of our
% paper. The code for balancing Bayesian inference is taken from
% https://github.com/elizqian/balancing-bayesian-inference.

clear; close all
addpath('models')
addpath('external_functions')

%% define LTI model
% heat equation
load('heat-cont.mat');
d           = size(A,1);
B           = eye(d);
sig_obs     = 0.008;

d_out       = size(C,1);

%% compute compatible prior = Reachability Gramian
L_pr        = lyapchol(A,B)';  
Gamma_pr    = L_pr*L_pr';
% draw random initial condition
x0          = L_pr*randn(d,1);
% generate random data from multiple initial conditions
num_reps    = 100;
x0_all      = L_pr*randn(d,num_reps);

%% compute infinite Obs Gramian
% helper matrix    
F           = C./sig_obs;
L_Q         = lyapchol(A',F')';
Q_inf       = L_Q*L_Q';

%% define time frame for inference problem
% measurement times and noise scaling
T           = [1,3,10];
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

    %% compute time-limited Obs Gramian
    Q_TL                = time_lim_Gramian_full(A',F',T(t));
    % this is done by a rational Krylov approach
    [~,~,~,~,ST,YST]    = RKSM_tlbt(A',M',F',5,[tol,tol,eps],maxit,'convR',[],[],T(t),[],[],opts);
    Q_TL_approx         = ST*YST*ST';
    % compare the approximated and the full time-limited Gramian
    rel_diff_Q_TL       = norm(Q_TL - Q_TL_approx)/norm(Q_TL)
    % compute a square root factorization of Q_TL
    % floating point computation errors induce complex zeros
    [V,D]               = eig(Q_TL_approx);
    V                   = real(V);
    E                   = real(sqrt(real(D)));
    L_Q_TL              = V*E;

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

    %% generate measurements
    % single measurement
    y       = G*x0;
    m       = y + sig_obs_long.*randn(n(t)*d_out,1);
    % multiple measurements for Bayes risk
    y_all   = G*x0_all;
    m_all   = y_all + sig_obs_long.*randn(n(t)*d_out,num_reps);

    %% compute true posterior
    full_rhs        = G'*(m./(sig_obs_long.^2));
    full_rhs_all    = G'*(m_all./(sig_obs_long.^2));

    L_prinv         = inv(L_pr); 
 
    R_posinv        = qr([Go; L_prinv],0);
    R_posinv        = triu(R_posinv(1:d,:)); % Pull out upper triangular factor
    R_pos_true      = inv(R_posinv);
    mupos_true      = R_posinv\(R_posinv'\full_rhs);
    mupos_true_all  = R_posinv\(R_posinv'\full_rhs_all);

    %% compute posterior approximations and errors
    r_vals      = 1:20;
    rmax        = max(r_vals);

    % (H,Gamma_pr^-1) computations
    [~,R]       = qr(Go,0);     % compute a square root factorization of H
    LG          = R';
    [V,S,W]     = svd(LG'*L_pr,0);
    tau         = diag(S);
    What        = L_pr*W;       % spantini directions
    Wtilde      = L_pr'\W;
    S           = S(1:rmax,1:rmax);   
    delH        = diag(S);
    Siginvsqrt  = diag(1./sqrt(delH));
    SrH         = (Siginvsqrt*V(:,1:rmax)'*LG')';
    TrH         = L_pr*W(:,1:rmax)*Siginvsqrt; % balancing transformation
    A_BTH       = SrH'*A*TrH;
    C_BTH       = C*TrH;

    %% balancing with Q_infty
    [V,S,W]     = svd(L_Q'*L_pr); 
    S           = S(1:rmax,1:rmax);
    delQ        = diag(S);
    Siginvsqrt  = diag(1./sqrt(delQ));
    Sr          = (Siginvsqrt*V(:,1:rmax)'*L_Q')';
    Tr          = L_pr*W(:,1:rmax)*Siginvsqrt; % balancing transformation
    A_BTQ       = Sr'*A*Tr;
    C_BTQ       = C*Tr;

    %% time-limited balancing with Q_TL 
    [V,S,W]     = svd(L_Q_TL'*L_pr); 
    S           = S(1:rmax,1:rmax);
    delQ_TL     = diag(S);
    Siginvsqrt  = diag(1./sqrt(delQ_TL));
    Sr_TL       = (Siginvsqrt*V(:,1:rmax)'*L_Q_TL')';
    Tr_TL       = L_pr*W(:,1:rmax)*Siginvsqrt; % balancing transformation
    A_BTQ_TL    = Sr_TL'*A*Tr_TL;
    C_BTQ_TL    = C*Tr_TL;

    %% compute posterior approximations
    f_dist                                      = zeros(length(r_vals),4);
    [mu_LRU, mu_LR, mu_BTQ, mu_BTH, mu_BTQ_TL]  = deal(zeros(d,length(r_vals)));
    mu_errs                                     = zeros(length(r_vals),5);
    for rr = 1:length(r_vals)
        r           = r_vals(rr);
        
        %% Spantini posterior quantities
        % Spantini approx posterior covariance
        Rpos_sp         = What*diag(sqrt([1./(1+tau(1:r).^2); ones(d-r,1)]));
        Gpos_sp=What*diag([1./(1+tau(1:r).^2); ones(d-r,1)])*What';

        f_dist(rr,1)    = forstner(Rpos_sp,R_pos_true,'sqrt');
    
        % Spantini approx posterior means
        Pi_r            = What(:,1:r)*Wtilde(:,1:r)';
        temp_sp_1       = Gpos_sp*Pi_r'*full_rhs_all;
        temp_sp_1       = R_pos_true\(temp_sp_1 - mupos_true_all);
        mu_errs(rr,1)   = mean(sqrt(sum(temp_sp_1.^2)));
    
        temp_sp_2       = Gpos_sp*full_rhs_all;
        temp_sp_2       = R_pos_true\(temp_sp_2-mupos_true_all);
        mu_errs(rr,2)   = mean(sqrt(sum(temp_sp_2.^2)));
    
        %% Q_infty posterior quantities
        % Balancing with Q_infty - generate G_BT,H_BT
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
    
        f_dist(rr,2)    = forstner(R_pos_BTQ,R_pos_true,'sqrt');
        temp_BTQ        = Gpos_BTQ*G_BTQ'*(m_all./(sig_obs_long.^2));
        temp_BTQ        = R_pos_true\(temp_BTQ - mupos_true_all);
        mu_errs(rr,3)   = mean(sqrt(sum(temp_BTQ.^2)));

    %% Q_TL time-limited posterior quantities
        % Balancing with Q_TL - generate G_BT,H_BT
        G_BTQ_TL        = zeros(n(t)*d_out,r);
        iter            = expm(A_BTQ_TL(1:r,1:r)*dt_obs);
        temp            = C_BTQ_TL(:,1:r);
        for i = 1:n(t)
            temp                                = temp*iter;
            G_BTQ_TL((i-1)*d_out+1:i*d_out,:)   = temp;
        end
        G_BTQ_TL        = G_BTQ_TL*Sr_TL(:,1:r)';
        G_BTQo_TL       = G_BTQ_TL./sig_obs_long;
        H_BTQ_TL        = G_BTQo_TL'*G_BTQo_TL;

        % Balancing with Q_TL - compute posterior covariance and mean
        R_posinv        = qr([G_BTQo_TL; L_prinv],0);
        R_posinv        = triu(R_posinv(1:d,:)); % Pull out upper triangular factor
        R_pos_BTQ_TL    = inv(R_posinv);
        Gpos_BTQ_TL     = R_pos_BTQ_TL*R_pos_BTQ_TL';
    
        f_dist(rr,4)    = forstner(R_pos_BTQ_TL,R_pos_true,'sqrt');
        temp_BTQ_TL     = Gpos_BTQ_TL*G_BTQ_TL'*(m_all./(sig_obs_long.^2));
        temp_BTQ_TL     = R_pos_true\(temp_BTQ_TL - mupos_true_all);
        mu_errs(rr,5)   = mean(sqrt(sum(temp_BTQ_TL.^2)));

        %% H posterior quantities
        % Balancing with H - generate G_BT, H_BT
        G_BTH           = zeros(n(t)*d_out,r);
        iter            = expm(A_BTH(1:r,1:r)*dt_obs);
        temp            = C_BTH(:,1:r);
        for i = 1:n(t)
            temp                            = temp*iter;
            G_BTH((i-1)*d_out+1:i*d_out,:)  = temp;
        end
        G_BTH           = G_BTH*SrH(:,1:r)';
        G_BTHo          = G_BTH./sig_obs_long;
        H_BTH           = G_BTHo'*G_BTHo;

        % Balancing with H - compute posterior covariance and mean
        R_posinv        = qr([G_BTHo; L_prinv],0);
        R_posinv        = triu(R_posinv(1:d,:)); % Pull out upper triangular factor
        R_pos_BTH       = inv(R_posinv);
        Gpos_BTH        = R_pos_BTH*R_pos_BTH';
       
        f_dist(rr,3)    = forstner(R_pos_BTH,R_pos_true,'sqrt');
        temp_BTH        = Gpos_BTH*G_BTH'*(m_all./(sig_obs_long.^2));
        temp_BTH        = R_pos_true\(temp_BTH - mupos_true_all);
        mu_errs(rr,4)   = mean(sqrt(sum(temp_BTH.^2)));
    end

    %% plots
    % Warning if complex parts of Förstner distances are nontrivial
    if ~isempty(find(abs(imag(f_dist))>eps*abs(real(f_dist)), 1))
        warning('Significant imaginary parts found in Forstner distance')
    end
    % Otherwise imaginary parts are trivial artifacts of generalized eig
    f_dist  = real(f_dist);
 
    %% plot posterior covariance Forstner errors
    subplot(T_length,3,3*t)
    semilogy(r_vals,f_dist(:,2),'o','Color',[0.8500    0.3250    0.0980]); hold on
    semilogy(r_vals,f_dist(:,4),'*','Color','magenta');
    semilogy(r_vals,f_dist(:,1),'Color','blue'); 
%     semilogy(r_vals,f_dist(:,3),'x','Color','#7E2F8E');
%     if you add this for BT-H, add 'BT-H' to the legend
    grid on
    if t == 1
        title('$\Gamma_{pos}$ F\"orstner error','interpreter','latex','fontsize',20)    
    end
    if t == T_length
        xlabel('$r$','interpreter','latex','fontsize',13)
        legend({'BT','TLBT','OLR'},'interpreter','latex','fontsize',13,'Location','southwest')
        legend boxoff
    end
    ylim([1e-15 1000])
    set(gca,'fontsize',13,'ticklabelinterpreter','latex')

    %% plot posterior mean errors in Gpos^-1 norm
    posnormref = mean(sqrt(sum((R_pos_true\mupos_true_all).^2)));
    subplot(T_length,3,3*t-1)
    semilogy(r_vals,mu_errs(:,3)/posnormref,'o','Color',[0.8500    0.3250    0.0980]); hold on
    semilogy(r_vals,mu_errs(:,5)/posnormref,'*','Color','magenta');
    semilogy(r_vals,mu_errs(:,1)/posnormref,'Color','blue'); 
%     semilogy(r_vals,mu_errs(:,2)/posnormref,'Color','cyan');
%     if you add this for OLRU, add 'OLRU' to the legend
%     semilogy(r_vals,mu_errs(:,4)/posnormref,'x','Color','#7E2F8E');
%     if you add this for BT-H, add 'BT-H' to the legend
    grid on
   if t == 1 
       title('$\mu_{pos}$  Bayes risk','interpreter','latex','fontsize',20)
    end
    if t == T_length
         xlabel('$r$','interpreter','latex','fontsize',13)
        legend({'BT','TLBT','OLR'},'interpreter','latex','fontsize',13,'Location','southwest')
        legend boxoff
    end
    ylim([1e-10 1])
    set(gca,'fontsize',13,'ticklabelinterpreter','latex')

    %% plot HSVs
    subplot(T_length,3,3*t-2)
    semilogy(r_vals,delQ/delQ(1),'o','Color',[0.8500    0.3250    0.0980]); hold on
    semilogy(r_vals,delQ_TL/delQ_TL(1),'*','Color','magenta'); hold on
%     semilogy(r_vals,delH/delH(1),'x','Color','#7E2F8E'); hold on
%     if you add this for BT-H, add '$\tau_i$' to the legend
    grid on
    if t == 1
        title('HSVs','interpreter','latex','fontsize',20)
    end
    if t == T_length
        xlabel('$i$','interpreter','latex','fontsize',13)
        legend({'$\delta_i$','$\delta^{TL}_i$'},'interpreter','latex','fontsize',13,'Location','southwest')
        legend boxoff
    end
    ylabel(['$t_e = $ ', num2str(T(t))],'interpreter','latex','fontsize',20)
    ylim([1e-10 1])
    set(gca,'fontsize',13,'ticklabelinterpreter','latex')
end
