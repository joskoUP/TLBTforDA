% This is a script to create the advection-diffusion equation example
% (Figure 4) of our paper. The code for balancing Bayesian inference is
% taken from https://github.com/elizqian/balancing-bayesian-inference.

clear; close all
addpath('external_functions')

%% define LTI model
% advection diffusion equation
d           = 200;
DK          = 0.02; % diffusion coefficent
u           = 0.01; % constant mean flow
entry       = u*d - DK*d*d;
main_diag   = [entry, (2*entry).*ones(1,d-1)];
upper_diag  = (-entry).* ones(1,d-1);
lower_diag  = (DK*d*d).*ones(1,d-1);
A           = diag(main_diag) + diag(upper_diag,1) + diag(lower_diag,-1);
% unstable matrix so can't compute infinite Gramians

sig_obs     = 0.008;
C           = ones(1,d)/d;
% helper matrix    
F           = C./sig_obs;
d_out       = size(C,1);

%% set given prior = Reach abilityGramian
L_pr        = eye(d);
Gamma_pr    = L_pr*L_pr';
M           = A*Gamma_pr+Gamma_pr*A';
if sum(real(eig(M))>0) > 0
    disp('The prior covariance L_pr is not prior-compatible.')
end
% draw random initial condition
x0          = L_pr*randn(d,1);
% generate random data from multiple initial conditions
num_reps    = 100;
x0_all      = L_pr*randn(d,num_reps);

%% define time frame for inference problem
% measurement times and noise scaling
T           = [0.1,0.5,1];
T_length    = length(T);
dt_obs      = 1e-3;
% for unstable A, Gramian computation directly via integration
fun         = @(t) expm(A'*t)*(C'*C)*expm(A*t);
% making dt_obs bigger makes Spantini eigvals decay faster
n           = round(T/dt_obs);
% relative noise scaling
scl_sig_obs = 0.1;   

figure(1); clf  
for t = 1:T_length
    if ~exist('sig_obs','var')
        sig_obs = scl_sig_obs*max(abs(reshape(y,d_out,n(t))),[],2);
    end
    sig_obs_long = repmat(sig_obs,n(t),1);

    %% compute time-limited Obs Gramian
    Q_TL                = integral(fun,0,T(t),'ArrayValued',true);
    % compute a square root factorization of Q_inf
    % floating point computation errors induce complex zeros
    [V,D]               = eig(Q_TL);
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
    r_vals      = 1:30;
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

    %% balancing with Q_infty time-limited
    [V,S,W]     = svd(L_Q_TL'*L_pr); 
    S           = S(1:rmax,1:rmax);
    delQ_TL     = diag(S);
    Siginvsqrt  = diag(1./sqrt(delQ_TL));
    Sr_TL       = (Siginvsqrt*V(:,1:rmax)'*L_Q_TL')';
    Tr_TL       = L_pr*W(:,1:rmax)*Siginvsqrt; % balancing transformation
    A_BTQ_TL    = Sr_TL'*A*Tr_TL;
    C_BTQ_TL    = C*Tr_TL;

    %% compute posterior approximations
    f_dist                                      = zeros(length(r_vals),3);
    mu_errs                                     = zeros(length(r_vals),4);
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
    
    %% Q_infty time-limited posterior quantities
        % Balancing with Q_infty - generate G_BT,H_BT
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

        % Balancing with Q_infty - compute posterior covariance and mean
        R_posinv        = qr([G_BTQo_TL; L_prinv],0);
        R_posinv        = triu(R_posinv(1:d,:)); % Pull out upper triangular factor
        R_pos_BTQ_TL    = inv(R_posinv);
        Gpos_BTQ_TL     = R_pos_BTQ_TL*R_pos_BTQ_TL';
    
        f_dist(rr,2)    = forstner(R_pos_BTQ_TL,R_pos_true,'sqrt');
        temp_BTQ_TL     = Gpos_BTQ_TL*G_BTQ_TL'*(m_all./(sig_obs_long.^2));
        temp_BTQ_TL     = R_pos_true\(temp_BTQ_TL - mupos_true_all);
        mu_errs(rr,3)   = mean(sqrt(sum(temp_BTQ_TL.^2)));

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
    semilogy(r_vals,f_dist(:,3),'x','Color','#7E2F8E'); hold on
    semilogy(r_vals,f_dist(:,2),'*','Color','magenta');
    semilogy(r_vals,f_dist(:,1),'Color','blue');
    grid on
    if t == 1
        title('$\Gamma_{pos}$ F\"orstner error','interpreter','latex','fontsize',20)    
    end
    if t == T_length
        xlabel('$r$','interpreter','latex','fontsize',13)
        legend({'BT-H','TLBT','OLR'},'interpreter','latex','fontsize',13,'Location','southwest')
        legend boxoff
    end
    xlim([0,rmax])
    ylim([1e-25 100])
    set(gca,'fontsize',13,'ticklabelinterpreter','latex')

    %% plot posterior mean errors in Gpos^-1 norm
    posnormref = mean(sqrt(sum((R_pos_true\mupos_true_all).^2)));
    subplot(T_length,3,3*t-1)
    semilogy(r_vals,mu_errs(:,4)/posnormref,'x','Color','#7E2F8E'); hold on
%     semilogy(r_vals,mu_errs(:,2)/posnormref,'Color','cyan');
%     if you add this for OLRU, add 'OLRU' to the legend
    semilogy(r_vals,mu_errs(:,3)/posnormref,'*','Color','magenta');
    semilogy(r_vals,mu_errs(:,1)/posnormref,'Color','blue');
    grid on
   if t == 1 
       title('$\mu_{pos}$  Bayes risk','interpreter','latex','fontsize',20)
    end
    if t == T_length
         xlabel('$r$','interpreter','latex','fontsize',13)
        legend({'BT-H','TLBT', 'OLR'},'interpreter','latex','fontsize',13,'Location','southwest')
        legend boxoff
    end
    xlim([0,rmax])
    ylim([1e-10 1])
    set(gca,'fontsize',13,'ticklabelinterpreter','latex')

    %% plot HSVs
    subplot(T_length,3,3*t-2)
    semilogy(r_vals,delQ_TL/delQ_TL(1),'*','Color','magenta'); hold on
    semilogy(r_vals,delH/delH(1),'x','Color','#7E2F8E'); hold on
    grid on
    if t == 1
        title('HSVs','interpreter','latex','fontsize',20)
    end
    if t == T_length
        xlabel('$i$','interpreter','latex','fontsize',13)
        legend({'$\delta^{TL}_i$','$\tau_i$'},'interpreter','latex','fontsize',13,'Location','southwest')
        legend boxoff
    end
    ylabel(['$t_e = $ ', num2str(T(t))],'interpreter','latex','fontsize',20)
    xlim([0,rmax])
    ylim([1e-20 1])
    set(gca,'fontsize',13,'ticklabelinterpreter','latex')
end