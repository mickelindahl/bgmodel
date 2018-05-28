function delta_v = voltage_finder(V, trg, rev_syn)
    delta_v = zeros(size(trg));
    for trg_ind = 1:length(trg)
        f_ind = V(:,1)==trg(trg_ind);
        delta_v(trg_ind) = V(f_ind,3) - rev_syn;
    end
end