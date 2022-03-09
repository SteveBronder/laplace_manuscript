
functions {
  matrix K_functor (vector[] x, int n_obs, real alpha, real rho) {
    matrix[n_obs, n_obs] K = cov_exp_quad(x, alpha, rho);
    for (i in 1:n_obs) K[i, i] += 1e-8;
    return K;
  }
}

data {
  int n_obs;
  int n_covariates;
  int y[n_obs];
  vector[n_obs] ye;
  array[n_obs] vector[n_covariates] x;
  real rho_alpha_prior;
  real rho_beta_prior;
}

transformed data {
  real tol = 1e-6;
  int max_num_steps = 100;
  vector[n_obs] theta_0 = rep_vector(0, n_obs);
  int n_samples[n_obs] = rep_array(1, n_obs);
}

parameters {
  real<lower = 0> alpha;
  real<lower = 0> rho;
}

model {
  rho ~ inv_gamma(rho_alpha_prior, rho_beta_prior);
  alpha ~ inv_gamma(10, 10);

  target += laplace_marginal_poisson_log_lpmf(y | n_samples, ye, theta_0, K_functor,
                                      x, n_obs, alpha, rho);
}

generated quantities {
  vector[n_obs] theta
    = laplace_marginal_poisson_log_rng(y, n_samples, ye, theta_0, K_functor,
       forward_as_tuple(x, n_obs), forward_as_tuple(x, n_obs), alpha, rho);
}
