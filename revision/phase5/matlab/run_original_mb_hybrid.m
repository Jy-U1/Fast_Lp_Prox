% Run the ORIGINAL Matlab evaluator from [24] on an independent tiny probe set.
% NOTE: This uses the source file xplus_lp_hybrid.m from
% https://github.com/lshen03/Lp-Proximity-Operator
% This MATLAB experiment cannot be directly compared with A100 PyTorch
% throughput; it addresses numerical behavior and implementation cost.
%
% Set env vars before launch (macOS / Linux):
%   export FPLP_ORIGINAL_MATLAB_REPO=/absolute/path/Lp-Proximity-Operator
%   export FPLP_MB_PROBES_FILE=/absolute/path/phase5_matlab_probes.csv
%   export FPLP_MB_OUTPUT_FILE=/absolute/path/phase5_original_mb_results.csv
%
source_dir = getenv('FPLP_ORIGINAL_MATLAB_REPO');
input_file = getenv('FPLP_MB_PROBES_FILE');
output_file = getenv('FPLP_MB_OUTPUT_FILE');
assert(~isempty(source_dir) && isfolder(source_dir), 'Missing original MATLAB repository');
assert(~isempty(input_file) && isfile(input_file), 'Missing probe CSV');
assert(~isempty(output_file) && ~isfile(output_file), 'Refusing to overwrite output');
addpath(source_dir);
assert(exist('xplus_lp_hybrid','file')==2, 'Cannot find original xplus_lp_hybrid.m');

A = readtable(input_file);
N = height(A);
observed_x = nan(N,1);
abs_error = nan(N,1);
stationarity_residual = nan(N,1);
runtime_ms = nan(N,1);
reported_total_bound = nan(N,1);
reported_bound_under_target = false(N,1);
state = strings(N,1);
for i = 1:N
    p = A.p(i); lam = A.lam(i); y = A.y(i);
    ustar = (1-p)^(1-p) / (2-p)^(2-p);
    rho = (2*lam*(1-p))^(1/(2-p));
    tau = rho*(2-p)/(2*(1-p));
    eps_thr = lam*p*tau^(p-2);
    options = struct();
    options.eta = 1e-6;
    % Ensure the whole active domain of this p is admissible (unlike R=.9*u_star).
    options.R = 0.5*(ustar+eps_thr);
    options.maxN = 100;
    options.maxT = 1500;
    options.nt = 600;
    options.verbose = false;
    t0 = tic;
    try
        [xh, info] = xplus_lp_hybrid(y,p,lam,options);
        runtime_ms(i) = 1000*toc(t0);
        observed_x(i) = real(xh);
        abs_error(i) = abs(real(xh)-A.ref_x(i));
        if isfinite(real(xh)) && real(xh)>0
            stationarity_residual(i) = abs(real(xh)-y+lam*p*real(xh)^(p-1));
        end
        if isfield(info,'total_bound')
            reported_total_bound(i) = info.total_bound;
            reported_bound_under_target(i) = (info.total_bound <= options.eta);
        end
        state(i) = "completed_without_float_certificate";
    catch ME
        runtime_ms(i) = 1000*toc(t0);
        state(i) = "failed: " + string(ME.identifier);
    end
    fprintf('%d/%d: p=%.5g factor=%.5g status=%s time_ms=%.3f\n', ...
        i,N,p,A.factor(i),char(state(i)),runtime_ms(i));
end
A.observed_x = observed_x;
A.abs_error = abs_error;
A.stationarity_residual = stationarity_residual;
A.runtime_ms = runtime_ms;
A.reported_total_bound = reported_total_bound;
A.reported_bound_under_target = reported_bound_under_target;
A.status = state;
writetable(A,output_file);
fprintf('Wrote original-MATLAB numerical comparison: %s\n',output_file);
% IMPORTANT: Reported bounds in the historic MATLAB routine include sampled
% numerical envelopes and do not include validated quadrature rounding error.
% Their reported values must not be called certified floating-point bounds.
