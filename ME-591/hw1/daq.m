function y = daq(v_ref_high, v_ref_low, n_bits, signal_v)
v_ref_high = v_ref_high + abs(v_ref_low);
resolution = (2^n_bits - 1);
delta = v_ref_high / resolution;
binary = zeros(resolution, 1);
signal_v = signal_v + abs(v_ref_low);
for i=1:resolution
    comparator = v_ref_high - i * delta;
    bit = signal_v >= comparator;
    binary(i) = (bit * comparator);
    if bit
        signal_v = signal_v - comparator;
    end
end
y = sum(binary) - abs(v_ref_low);
