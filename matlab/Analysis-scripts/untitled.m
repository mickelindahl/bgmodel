A = unique(params(:,1));
B = find(params(:,1)==A(1));
C = diff(B);
D = find(C>1);
E = [];
for ii = 1:length(D)+1
    E = [E;500*ii*ones(D(1)*length(A),1)];
end