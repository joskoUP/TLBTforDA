% function RKSM_TLBT.m
% Many thanks to Patrick Kürschner for providing this Code.

function [funAb,xi,err,V,Z,Y,beta] = RKSM_tlbt(A,E,B,sfreq, tol,maxit,sh,exact,npts,T,ssolve,modG,opts)
% Rational Krylov method for computing low-rank solution factors of the
% solution of the time-limited Gramian equation
%   AXE'+EXA-yy'+BB'=0, where
%
% y = exp(A*T)*B
%
%  CALLING SEQUENCE:
%     [funAb,xi,err,V,Z,Y,beta] = RKSM_tlbt(A,E,v,sfreq, tol,maxit,sh,exact,npts,T,ssolve,modG,opts)
%
% INPUT:
%    E,A    -     n-x-n matrices
%    B      -     n-x-m matrix (m << n);
%    sfreq  -    frequency of small-scale solution: reduced expm(..)x and GCALE is computed/solved every sfreq iteration
%    tol    -     stopping tolerance(s) w.r.t. 
%                   tol(1): y=f(A)B, tol(2) GCALE residual (default tol(1)), 
%                   and tol(3) rank truncation in the end (default 0)
%    maxit  -     maximal number iterations=> maxdim=(maxit+-1)*m
%    sh     -    shifts (poles) for RKSM (default emptry-> adaptive shifts)
%		can be a vector of precomputed shifts, or
%               cans be a string for the adaptive shifts:
%               'conv' -- choose from convex hull of Ritz values
%               'convR' -- real version of 'conv'
%               'real' -- uses predetermined real interval (outdated)
%    exact  -    precomputed exact f(A)*B for testing        
%    npts   -     numbers of test points for adaptive shift selection
%    T    -     finite time value
%    ssolve  - type of small y=f(A)B handling, currently not used
%    modG    - set to 1 for modified time-limited Gramians (rhs is transformed [B,f(A)B]) 
% opts    = structure containing info on mass matrix handling 
%           opts.ML,opts.MU -- precomputed Cholesky or LU factors of E=ML*MU, if E=E^T,
%           MU=ML'
%           opts.facM - compute factors of E if empty ML,MU, 
%               facM=1: compute Chol. / LU factors of E 
%               facM=2: compute Chol. / LU factors of E with pivoting, DEFAULT  
%               facM=0: or directly work with E (facM=0)

% OUTPUT:
%    funAb      approx of exp(A*T)B (or generalized state-space equivalent)
%    xi         generated / used shifts   
%    err       vector containing 
%		1-row: iteration indices for small-scale solve
%		2nd row: the relative changes ||ynew-yold||/||ynew||, uses exact if given
%		3rd row: Lyap. res.norm
%    V         orthonormal basis matrix of the rational Krylov subspace
%    Z,Y       low-rank solution factors of GCALE solution ZYZ'~X
%    beta      orthogonalization factors of B 
%
%-----------------------------------------------------------------------
% P. Kuerschner, 2015-17
if nargin < 13 || isempty(opts), 
    opts.facM=2; opts.ML=[]; opts.MU=[];
end
if nargin < 12 || isempty(modG), modG=0; end
if nargin < 9 || isempty(npts), npts = 1601; end
if nargin < 8 || isempty(exact), exact = []; end;
if nargin < 7 || isempty(sh), 
    sh='im'; adaptive = 1;%xi_cand = 1i*logspace(log10(f(1)),log10(f(2)),npts);
end
if size(tol,2)>2 % stopping tolerances
    tolP=tol(2);
    tolsol=tol(3);
    tol=tol(1);
else tolP=tol; 
    tolsol=0;
end

if nargin<11 || isempty(ssolve) % currently not used
    ssolve='expm';
end

% shift settings
if isnumeric(sh)
    xi=sh;adaptive = 0;
else
    switch(sh)
        case 'conv'
            adaptive = 1; xi=[];
        case 'convR'
            adaptive = 1; xi=[];
        case 'real'
            adaptive = 1; xi=[];
            xi_cand = logspace(-10,10,npts);
        otherwise
            adaptive = 1; xi=[]; sh='conv';
    end;
end
[N,s] = size(B);
if (norm(A-A',1)<1e-14), symA=1; else symA=0;end % is A=A'?
% handling of mass matrix
if isempty(E) 
    E=speye(N); ML=speye(N); MU=ML;  pM=1:N; qM=1:N;
else 
    if (norm(E-E',1)<1e-14), symM=1; else symM=0;end % is E=E'?
    if ~isempty(opts.ML)
      ML=opts.ML; 
      if ~isempty(opts.MU), MU=opts.MU;  
      else
          if symM, MU=ML'; else MU=speye(N); end
      end
    else
        switch opts.facM
            case 1 % use LU / Cholesky factorization
                ls=tic;
                if symM
                    ML=chol(E,'lower'); MU=ML';
                else
                    [ML,MU]=lu(E);
                end
                pM=1:N; qM=1:N;
                timings(1,1)=toc(ls);
            case 2 % use LU / Cholesky factorization with pivoting
                ls=tic;
                if symM
                    [ML,~,pM]=chol(E,'lower','vector'); MU=ML';
                    qM(pM)=1:N;
                else
                    [ML,MU,pM,qM]=lu(E,'vector');
                end
                timings(1,1)=toc(ls);
            otherwise %no factor.
                ML=E;
                MU=speye(N); pM=1:N; qM=1:N;
        end
    end
%     nML=normest(ML);
end
if length(T)>1, if min(T)==0, T=max(T); end, end
if ~isempty(exact), 
    switch opts.facM,
        case 2
            exact=full(ML\exact(pM,:));
        otherwise            
            exact=full(ML\exact);
    end
    Nexact=norm(exact); 
end


if modG
    flip=blkdiag(eye(s),-eye(s));
    modG_rhs=0;
end
Is=speye(s);nrmrestot=[];
fAb_conv=0;


if nargin < 4 || isempty(maxit), maxit = min(N,50); end;
if nargin < 3 || isempty(tol), tol = 1e-8; end;
solver = @(A,b) (-A)\(-b);    % can use your favorite solver here
zeta = 1*zeros(maxit,1);   % may be optimized

% start RatArn process
switch opts.facM,
    case 1
        vE=ML\B;
    case 2
        vE=ML\B(pM,:);
    otherwise
        vE=ML\B;
end

vEo=vE;
nu1 = s;
[vE,beta]=qr(vE,0);

s1=-0.1;
%         nA=eigs(A,E,1,'SR');
        nA=-normest(A);
if adaptive && (strcmp(sh,'conv') || strcmp(sh,'convR'))
%       try
%         s1=eigs(A,E,1,'LR');
%       catch
%          s1=real(eigs(A,E,1,'SM'));
%       end
      
%         s1=min(abs(ss));
%         nA=max(abs(ss));
%         normest(A);
        xi(1)=-s1;
end

V = zeros(N,s*2*maxit+1); 
AV=V;
V(:,1:nu1) = vE;
switch opts.facM  
    case 1
         AV(:,1:nu1) = ML\(A*(MU\vE)); 
    case 2
         w=  MU\vE;
         w = A*w(qM,:);
         AV(:,1:nu1) = ML\w(pM,:);
    otherwise
         AV(:,1:nu1) = ML\(A*vE);
end

Rt=vE'*AV(:,1:nu1);
theta=eig(Rt);
H = zeros(2*s*maxit+1,2*s*maxit);
I = speye(N);O=0*Is;
reorth = 1; 
fmR0 = []; if length(T)>1, fmR0a=[]; end
xi = xi(:).';
fmR=zeros(N,1);
n=1;jc=1; cplx=0;
curr_start = 1; n = 1; sigma = -1;
while n <= maxit, %n
    prev_start = curr_start;      % .. The first index where columns were added in the previous step.
        prev_end   = prev_start+nu1-1; % .. The  last index where columns were added in the previous step. [curr_start, curr_end]
        real_start = prev_end+1;      % .. The first index where columns (for the real part) will be added in the current step.
        real_end   = real_start+nu1-1; % .. The  last index where columns (for the real part) will be added in the current step. [real_start, real_end].
        imag_start = real_end+1;      % .. The first index where columns for the imaginary part will be added in the current step if the shift is complex.
        imag_end   = imag_start+nu1-1; % .. The  last index where columns for the imaginary part will be added in the current step if the shift is complex. [imag_start, imag_end].
   
    jms=(n-1)*s+1;j1s=(n+1)*s;js=n*s;js1=js+1;
    sigma_prev = sigma;
    if adaptive, %shift computation
        % look for minimum of nodal rational function sn
        theta(real(theta)>0)=-theta(real(theta)>0);
        if strcmp(sh,'convR') , theta=real(theta); end
        thetao=theta;
        if strcmp(sh,'conv') || strcmp(sh,'convR') %&& length(theta)>2 %
            %  convex hull test set ala
            %  Simoncini/Druskin
            if any(imag(theta)) && length(theta)>2
                theta=[theta;nA];
                ch=convhull(real(theta),imag(theta));
                eH=-theta(ch);
                ieH=length(eH); missing=n*s-ieH;
                while missing>0, % include enough points from the border
                    neweH=(eH(1:ieH-1)+eH(2:ieH))/2;
                    eH=[eH;neweH];
                    missing=n*s-length(eH);
                end
                
            else
                eH=sort([-real(theta);-s1;-nA]);
            end
            if n==1, eH=sort(-[s1;nA]); end
            xi_cand=[];
            for j=1:length(eH)-1
                xi_cand=[xi_cand,linspace(eH(j),eH(j+1),500/s)];
            end
        end
        if n==1
            if strcmp(sh,'conv') || strcmp(sh,'convR'), 
                gs=-s1*ones(s,1); 
            else, gs=inf*ones(s,1); 
            end
        else
            gs=kron([xi(2:end)],ones(1,s))';
        end
        sn=ratfun(xi_cand,thetao,gs);
        [~,jx]=max(abs(sn));
        if real(xi_cand(jx))<0, xi_cand(jx)=-xi_cand(jx); end
        if abs(imag(xi_cand(jx))) / abs( xi_cand(jx))<1e-8
            xi(n+1)=real(xi_cand(jx));
        else
            xi(n+1)=xi_cand(jx);
        end
        
    end;
    sigma=xi(n+adaptive);
%     if( sigma_prev == conj(sigma) )
%             % .. Complex conjugate shift, we did this one already, skip
%             continue;
%     end

    switch opts.facM  
        case 1
            w1 = MU*(solver(A - xi(n+adaptive)*E,ML*V(:, prev_start:prev_end)));
        case 2
            w1(pM,:)=ML*V(:, prev_start:prev_end);
            w1=solver(A - xi(n+adaptive)*E,w1);
            w1=MU*w1(pM,:);
        otherwise
            w1 = solver(A - xi(n+adaptive)*E,ML*V(:, prev_start:prev_end));
    end
    
    % real (!) orthonormal expansion of basis
    cplx=0;
    wR = real(w1);
    for it=1:2,
            for kk=1:n
                k1=(kk-1)*nu1+1; k2=kk*nu1;
                gamma=V(:,k1:k2)'*wR;
                H(k1:k2,real_start-nu1:real_end-nu1) = H(k1:k2,real_start-nu1:real_end-nu1)+ gamma;
                wR = wR - V(:,k1:k2)*gamma;
            end
    end
    [V(:,real_start:real_end),H(real_start:real_end, real_start-nu1:real_end-nu1)]=qr(wR,0); 
    if ~isreal(xi(n+adaptive))
        cplx=1;
        wR=imag(w1);
        for it=1:2,
            for kk=1:n+1
                k1=(kk-1)*s+1; k2=kk*s;
                gamma=V(:,k1:k2)'*wR;
                H(k1:k2,imag_start-nu1:imag_end-nu1) = H(k1:k2,imag_start-nu1:imag_end-nu1)+ gamma;
                wR = wR - V(:,k1:k2)*gamma;
            end
        end
        [V(:,imag_start:imag_end),H(imag_start:imag_end, imag_start-nu1:imag_end-nu1),]=qr(wR,0); 
        
        curr_start = real_start;  % .. Where the new columns in Z start
        curr_end   = imag_end;    % .. Where the new columns in Z end
        proj_end   = real_end;    % .. Where the columns used for projection end
    else
        cplx=0;
        % .. The shift is real
        curr_start = real_start;
        curr_end   = real_end;
        proj_end   = prev_end;
    end
    %reduced matrix, TODO: make this more efficient
    switch opts.facM  
        case 1
            AV(:,curr_start:curr_end)=ML\(A*(MU\V(:,curr_start:curr_end)));
        case 2
            w=  MU\V(:,curr_start:curr_end);
            w = A*w(qM,:);
            AV(:,curr_start:curr_end) = ML\w(pM,:);
        otherwise
            AV(:,curr_start:curr_end)=ML\(A*(V(:,curr_start:curr_end)));
    end
    g  = V(:, 1:prev_end)' * AV(:, curr_start:curr_end);
    g3 = V(:, curr_start:curr_end)' * AV(:, 1:curr_end);
    Rt=[Rt,g; g3]; % update projected A

% if( ~isreal( sigma ) )
%             % Two step were done at once.
%             n = n + 1;
%             curr_start      = imag_start;
%             if adaptive, xi(n+1)=conj(xi(n)); end
% end
% 

justsolved=0;
% solve Galerkin system every sfreq steps, but 
% not in between a pair of compl. conj. shifts
cplxssolve=n>1 && ((xi(n-1)==conj(xi(n))) && mod(n-1,sfreq)==0);
if (mod(n,sfreq)==0)|| cplxssolve%% small scale solution
    justsolved=1;
    err(1,jc) = j1s-s;
    if ~fAb_conv
        %%compute small solution
        % time1
        fmR=expm(Rt(1:proj_end, 1:proj_end)*max(T))*[beta;zeros(proj_end-nu1,s)]; 
        df=size(fmR,1)-size(fmR0,1);
        if ~isempty(exact), % rel. change
            err(2,jc) = norm(V(:,1:proj_end)*fmR - exact)/Nexact;
        else %rel. change
%             err(2,jc) = norm([fmR0;zeros(df,s)]-fmR)/norm(fmR);
            err(2,jc) = norm(blkdiag(fmR0*fmR0',zeros(df,df))-fmR*fmR')/norm(fmR'*fmR);
        end;
        if length(T)>1% lower time bound>0
            fmRa=expm(Rt(1:proj_end, 1:proj_end)*min(T))*[beta;zeros(proj_end-nu1,s)];
            dfa=size(fmRa,1)-size(fmR0a,1);
            if ~isempty(exact), % rel. change
                err(2,jc) = norm(V(:,1:proj_end)*fmRa - exact)/Nexact;
            else %rel. change
                err(2,jc) = max(norm([fmR0a;zeros(df,s)]-fmRa)/norm(fmRa),err(2,jc));
            end;
        end
        
        fprintf('Step %d -- %e       %d        \n',n,err(2,jc),n*s);
        if err(2,jc) < tol,
            funAb_ss = V(:,1:proj_end)*fmR;  
            funAb(pM,:)=ML*funAb_ss;
            if length(T)>1, 
                funAb_ss_a = V(:,1:proj_end)*fmRa;  
                funAb_a(pM,:)=ML*funAb_ss_a;
            end     
            fAb_conv=1;
        end;
        fmR0=fmR; 
        if length(T)>1,  fmR0a=fmRa; end
    end
    if fAb_conv %f(A)B done, switch to GCALE sol.
        if tolP>0 %if required
            if ~adaptive || (~isnumeric(sh) && strcmp(sh,'im')), 
                sh='conv'; if symA && symM, sh='convR'; end
                adaptive=1; 
                xi=[1,xi(1:n)];
            end 
            if js==size(fmR,1)
                yt=fmR; %disp('drin')
                if length(T)>1, yta=fmRa; end
            else
                yt=V(:,1:proj_end)'*funAb_ss;%*[yr;zeros(2*m,m)];
                if length(T)>1, 
                    yta=V(:,1:proj_end)'*funAb_ss_a; 
                end
                %                 disp('nicht drin')
            end
            if modG % modified TL Gramian
                eigsops.issym=1; %eigs settings
                eigsops.tol=1e-10;
                eigsops.isreal=1;
                if ~modG_rhs % build the rhs factor of the modified fl CALEs
                    [Db,Eb]=eigs(@(x) [B,funAb]*(flip*([B,funAb]'*x)),N,2*s,'LM',eigsops);
                    e=diag(Eb);
                    [e,idx]=sort(abs(e),'descend');
                    % e=e(idx);
                    Db=Db(:,idx);
                    beta=(Db(:,1:2*s)*diag(e(1:2*s).^(.5)));
                    switch opts.facM 
                        case 1
                            Ba=ML\beta; 
                        case 2
                             Ba=ML\beta(pM,:);
                        otherwise
                            Ba=ML\beta; 
                    end
                    modG_rhs=1; funAb=beta;
                    if adaptive, sh='conv'; end
                end
                Ba_p=V(:,1:proj_end)'*Ba;
                rhs2=Ba_p*Ba_p';nBf=norm(rhs2);
            else
                if length(T)>1
                     rhs2=yta*yta'-yt*yt';
                else
                     roff=[beta;zeros(proj_end-nu1,s)];
                     rhs2=roff*roff'-yt*yt';
                end
                nBf=norm(rhs2);
            end
            % smale scale lyap. sol
            Y = lyap(Rt(1:proj_end, 1:proj_end),rhs2);
            % computed residual of large CALE (exact, in exact arithmetic)
            g1 = V(:, 1:curr_end-nu1)' * AV(:, curr_end-nu1+1:curr_end);
            u1 = AV(:, curr_end-nu1+1:curr_end) - V(:, 1:curr_end-nu1) * g1;
            
            g2 = V(:, curr_end-nu1+1:curr_end)' * u1; % p x p matrix
            u2 = u1 - V(:, curr_end-nu1+1:curr_end) * g2;
            
            g3 = qr( u2, 0 ); g3 = triu( g3(1:size(g3, 2), :) ); % p x p matrix; note Y is a proj_end x proj_end matrix, proj_end = k*nu1
            
            HT = H(curr_end-nu1+1:curr_end, proj_end-nu1+1:proj_end); % p x p matrix
            if( proj_end >= 2*nu1 )
                temp = [zeros(nu1, proj_end-nu1) HT] / H(1:proj_end, 1:proj_end); % nu1 x proj_end matrix
                R1 = ( [zeros(nu1, proj_end-2*nu1) -imag(sigma)*HT real(sigma)*HT] / H(1:proj_end, 1:proj_end) - g2*temp ) * Y;
                R2 = -g3*temp*Y;
            else
                temp = [zeros(nu1, proj_end-nu1) HT] / H(1:proj_end, 1:proj_end);
                R1 = ( (real(sigma)*HT) / H(1:proj_end, 1:proj_end) - g2*temp ) * Y;
                R2 = -g3*temp*Y;
            end
            err(3,jc)=norm( [R1; R2] ) / nBf;
            % abs(eigs(@(x) ML\(A*(ML'\((V(:,1:j1s-s)*Y)*(V(:,1:j1s-s)'*(x)))))+((V(:,1:j1s-s)*Y)*(V(:,1:j1s-s)'*(ML\(A'*(ML'\x)))))+B*(B'*x),N,1,'LM'))/(norm(B'*B,2))
            fprintf('Lyap: \t %d\t %e\t %d\n',n,err(3,jc),n*s)
            %             if modG
            %                              eigsops.tol=1e-2;
            %                  eigsops.maxit=10;
            %             abs(eigs(@(x) (ML\(A*((V(:,1:proj_end)*Y)*(V(:,1:proj_end)'*(x)))))+((V(:,1:proj_end)*Y)*(V(:,1:proj_end)'*((A'*(ML'\x)))))+vEo*(funAb_ss'*x)+funAb_ss*(vEo'*x),N,1,'LM',eigsops))/nBf
            %             end
            %disp([i,nrmres])
            if (err(3,jc)<tolP),
                % factored solution
                if tolsol % perform rank truncation
                    [uY,sY,~]=svd(Y); [sY,id]=sort(diag(sY),'descend');
                    uY=uY(:,id);
                    is=find(sY>tolsol*sY(1));
                    Y = uY(:,is)*diag(sqrt(sY(is)));
                    Z = V(:,1:size(Y,1))*Y;
                    switch opts.facM  
                        case 1
                            Z=MU\Z;
                        case 2
                            Z(pM,:) = MU\Z;
                    end
                    Y=eye(size(Z,2));
                else
                    switch opts.facM  
                        case 1
                            Z=MU\V(:,1:size(Y,1));
                        case 2
                            Z(pM,:) = MU\V(:,1:size(Y,1));
                        otherwise
                            Z=V(:,1:size(Y,1));
                    end
                end
                if length(T)>1 && ~modG,funAb=[funAb_a,funAb]; end
                V=V(:,1:proj_end);
                break,
                jc=jc+1;
            end
        else
            Z=[];
            V=V(:,1:proj_end);
            break;
        end
    end
    jc=jc+1;
end
if( ~isreal( sigma ) )
            % Two step were done at once.
            n = n + 1;
            curr_start      = imag_start;
            xi(n+1)=conj(xi(n));
end

if n >= maxit,
    if ~fAb_conv %mod(n,sfreq),
        fmR=V(:,1:proj_end)*expm(Rt(1:proj_end, 1:proj_end)*T)*[beta;zeros(proj_end-s,s)];
        switch opts.facM  
%                 case 1
%                       funAb=ML*fmR;
                case 2
                    funAb(pM,:)=ML*fmR;
                otherwise
                     funAb=ML*fmR;
        end
        Z=[];Y=[];
    else
        switch opts.facM  
            case 1
                Z=MU\V(:,1:size(Y,1));
            case 2
                Z(pM,:) = MU\V(:,1:size(Y,1));
            otherwise
                Z=V(:,1:size(Y,1));
        end
    end
    V=V(:,1:proj_end);
    disp(['Max. iteration number of ' num2str(maxit) ' reached.']);
    break
end;
% compute Ritz values for shift comp.
theta=eig(Rt(1:proj_end, 1:proj_end));

n=n+1;
end
       
function r=ratfun(x,eH,s)
% evaluate rational function
r=zeros(1,length(x));
for j=1:length(x)
    r(j)=abs(prod( (x(j)-s)./(x(j)-eH) ));
end       