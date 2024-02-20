%%%%%%%%%%%%%%%%%%%%%%%%%% IN THE NAME OFF GOD %%%%%%%%%%%%%%%%%%%%%%%%%%%%                                                         
%% finding det(F(beta ,z)) for different M:#nodes

for M = 4
    clc 
    clearvars -except M 
    close all
    syms z l mu M_sym eta beta rho alpha 
    
    I    = symmatrix(eye(M));
    Z    = symmatrix(zeros(M));
    Ones = symmatrix((1/M_sym)*ones(M,M));
    
    lmu2 = l*mu/2;
    S1 = (1-lmu2)*I - M_sym*eta*(I-Ones);
    S2 = -(1/2)*Ones + lmu2*I;
    S3 = (-mu/2)*I;
    S4 = -(1/2)*Ones + lmu2*I; 
    S5 = -lmu2*I;
    S6 = -(mu/2)*I;
    
    R1 = I;
    R2 = (-rho/mu)*I;
    R3 = (rho/mu)*Ones;
    R5 = alpha*I;
    R6 = (rho/mu)*I;
    R7 = (-rho/mu)*Ones;
    R9 = -alpha*I+Ones;

    F  = [z*S1, z*S2, z*S3, Z, I, -z*R1, -z*R2, -z*R6;
        z*S4, z*S5, Z,Z,Z, I, -z*R3, -z*R7;
        z*S6, Z,Z,Z,Z,Z, I, Z; 
        Z,Z,Z,Z,Z,Z, -z*R5, I-z*R9;
        z*I, Z,Z,Z, (-1/beta)*I, Z,Z,Z;
        -R1, z*I, Z,Z,Z,Z,Z,Z;
        -R2, -R3, (z-1)*I, -R5, Z,Z,Z,Z;
        -R6, -R7, Z, z*I-R9, Z,Z,Z,Z];
    
    detF = collect(symmatrix2sym(det(F)));
%     filename = "detF"+M+".mat";
%     save(filename, "detF", "M", '-mat')
end
%===============================================================================================
%===============================================================================================
%% finding roots, define M
% first, find a suitable step-size, rho and alpha, leading to real roots for all positive beta, given M,l,eta 
% second, for that choice of step-size, check combinations of alpha and rho, such that for all positive beta results in real roots. 

clc
clear
close all
digitsOld = digits(64);

syms z
M = 4;

filename = "detF"+M+".mat";
load(filename)

rho    = 1e-6;
alpha  = 0.5;

Beta   = logspace(-5,3,60);
Mu     = logspace(-5,4,60);
out1   = zeros(length(Beta), length(Mu));

i=0;
for beta = Beta
    i=i+1;
    disp(i)
    j=0;
    tic
    for mu = Mu
        j=j+1;
        true_roots = vpasolve(subs(detF),z);
        if isreal(true_roots)
            out1(i,j) = 1.0;
        end
    end
    toc
end

%%%% Correct two pixels that are mislabeled due to precision errors.
out1(81,89) = 1;
out1(82,84) = 1;
%===============================================================================================
%===============================================================================================
%% Plot the first figure in the paper
[x1,x2] = meshgrid(Mu, Beta);
filename = ['beta_mu_rho_' num2str(rho) '_alpha_' num2str(alpha) '.pdf'];

figure
[~,c] = contourf(x1,x2,out1,1);
c.LineWidth = 2;
colormap summer
set(gca,'xscale','log')
set(gca,'yscale','log')
set(gca,'FontSize',20, 'linewidth',2);
dummyh = line(nan, nan, 'Linestyle', 'none', 'Marker', 'o', 'Color', 'y');
leg = legend(dummyh, 'Feasible','Location','best');
set(leg,'FontName','Times New Roman','Interpreter','latex','FontSize',27);
xlabel('$\mathbf{\mu}$','FontSize',32,'Interpreter','latex');
ylabel('$\mathbf{\beta}$','FontSize',32, 'Interpreter','latex');
grid off;
xlim([0.8*min(Mu) 1.25*max(Mu)]);
ylim([0.8*min(Beta) 1.25*max(Beta)]);
xlim([min(Mu) max(Mu)]);
ylim([min(Beta) max(Beta)]);
saveas(gcf,filename)
%===============================================================================================
%===============================================================================================
%% second part

clc
clear
close all
digitsOld = digits(16);

syms z
M = 4;

filename = "detF"+M+".mat";
load(filename)

mu    = 1e-1;
eta   = 1e-1;
l     = 0.1;
M_sym = M;

Beta2  = logspace(-2,3,5);
Beta3  = logspace(0,10, 3);
Alpha2 = linspace(0,0.3,30);
Rho2   = logspace(-8,1,40);

out2   = zeros(length(Rho2) ,length(Alpha2));
thr    = 1e-5;
i=0;
for alpha = Alpha2
    i=i+1;
    disp(i)
    j=0;
    tic
    for rho = Rho2
        j=j+1;
        k=0;
        if alpha <1
            for beta = Beta2
                k=k+1;
                true_roots = vpasolve(subs(detF),z);
                if ~isreal(true_roots)
                    break;
                end
                if  k==length(Beta2)
                    out2(j,i)= 1;
                end
            end
        else
            for beta = Beta3
                k=k+1;
                true_roots = vpasolve(subs(detF),z);
                if ~isreal(true_roots)
                    break;
                end
                if  k==length(Beta2)
                    out2(j,i)= 1;
                end
            end
        end
    end
    toc
end
%===============================================================================================
%===============================================================================================
%% plot the second figure in the paper
[x3,x4] = meshgrid(Alpha2, Rho2);
filename = ['alpharho_mu_' num2str(mu) '_eta_' num2str(eta) '.pdf'];

figure
[M,c] = contourf(x3,x4,out2,1);
c.LineWidth = 2;
colormap summer %gray
% set(gca,'xscale','log')
set(gca,'yscale','log')
set(gca,'FontSize',20, 'linewidth',2);
dummyh = line(nan, nan, 'Linestyle', 'none', 'Marker', 'o', 'Color', 'y');
leg = legend(dummyh, 'Feasible','Location','best');
set(leg,'FontName','Times New Roman','Interpreter','latex','FontSize',27);
xlabel('$\alpha$','FontSize',32,'Interpreter','latex');
ylabel('$\rho$','FontSize',32, 'Interpreter','latex');
grid off;
xlim([min(Alpha2) max(Alpha2)]);
ylim([min(Rho2) max(Rho2)]);
saveas(gcf,filename)


