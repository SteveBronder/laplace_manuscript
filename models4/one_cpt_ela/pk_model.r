
rm(list = ls())
gc()
set.seed(1954)


#setwd("~/Code/laplace_approximation/Script/")
#.libPaths("~/Rlib")

# modelName <- "one_cpt"
modelName <- "one_cpt_ela"
# modelName <- "one_cpt_dense_cov"

scriptDir <- getwd()
modelDir <- file.path(scriptDir, "models4")
dataDir <- file.path(scriptDir, "data")
outDir <- file.path(scriptDir, "deliv", modelName)
delivDir <- file.path("deliv", modelName)

library(cmdstanr)
set_cmdstan_path("~/Code/laplace_approximation/spring2021/cmdStan/")
library(ggplot2)
library(rjson)

if (TRUE) {
  file <- file.path(modelDir, modelName, paste0(modelName, ".stan"))
  mod <- cmdstan_model(file)
}

# Simulate data
if (FALSE) {
  n_patients <- 10
  time_after_dose <- c(0.083, 0.167, 0.25, 1, 2, 4)  # c(0.167, 0.25, 2)
  n_measure <- length(time_after_dose)
  time <- rep(time_after_dose, n_patients)
  N <- n_patients * n_measure
  start <- seq(from = 1, to = N, by = n_measure)
  end <- seq(from = n_measure, to = N, by = n_measure)
  y0 <- 1000
  y_obs <- rnorm(N)

  stan_data <- list(N = N, y_obs = y_obs, time = time,
                    n_patients = n_patients, start = start, end = end,
                    y0 = y0)

  init_fixed <- function() {
    sigma_0 = 0.2
    sigma_1 = 0.2
    sigma = 1
    k_0_pop = 2
    k_1_pop = 1

    k_0 = rnorm(n_patients, k_0_pop, sigma_0)
    k_1 = rnorm(n_patients, k_1_pop, sigma_1)

    list(sigma_0 = sigma_0, sigma_1 = sigma_1, sigma = sigma,
         k_0_pop = k_0_pop, k_1_pop = k_1_pop, k_0 = k_0, k_1 = k_1)
  }

  fit <- mod$sample(
    fixed_param = TRUE,
    data = stan_data, init = init_fixed, chains = 1,
    iter_warmup = 0, iter_sampling = 1, seed = 123)

  y_pred <- c(fit$draws("y_pred"))
  patient_ID <- rep(1:n_patients, each = n_measure)
  p <- ggplot(data = data.frame(y = y_pred, x = time, ID = patient_ID),
              aes(y = y, x = x)) + theme_bw() + geom_point() +
    facet_wrap(~ID)
  p

  stan_data$y_obs <- y_pred

  write_stan_json(stan_data, paste0(dataDir, "/", modelName, ".data.json"))
}

# Fit data
stan_data <- fromJSON(file = paste0(dataDir, "/one_cpt.data.json"))

init <- function() {
  list(sigma_0 = abs(rnorm(1)),
       sigma_1 = abs(rnorm(1)),
       sigma = abs(rnorm(1)),
       k_0_pop =  rnorm(1, 2, 0.1),
       k_1_pop = rnorm(1, 1, 0.1),
       k_0 = rnorm(stan_data$n_patients, 2, 0.1),
       k_1 = rnorm(stan_data$n_patients, 1, 0.1))
}

fit <- mod$sample(data = stan_data, init = init,
                  chains = 4, parallel_chains = 4,
                  iter_warmup = 1000, iter_sampling = 1000,
                  seed = 123)
fit$summary()

# Try running model with better inits
draws <- fit$draws(variables = c("sigma_0", "sigma_1", "sigma", "k_0_pop",
                                 "k_1_pop", "k_0", "k_1"))[1, 1, ]

init2 <- function() {
  list(sigma_0 = draws[1],
       sigma_1 = draws[2],
       sigma = draws[3],
       k_0_pop = draws[4],
       k_1_pop = draws[5],
       k_0 = draws[6:(5 + stan_data$n_patients)],
       k_1 = draws[(6 + stan_data$n_patients):(5 + 2 * stan_data$n_patients)])
}

fit <- mod$sample(data = stan_data, init = init2,
                  chains = 4, parallel_chains = 4,
                  iter_warmup = 1000, iter_sampling = 1000,
                  seed = 123)

# Using inits from the sampling phase doesn't help much.
# Let's try using the warmed up covariance matrix.
inv_metric <- fit$inv_metric()
step_size <- fit$metadata()$step_size_adaptation

if (FALSE) {
  inv_metric$`1` <- diag(inv_metric$`1`)
  inv_metric$`2` <- diag(inv_metric$`2`)
  inv_metric$`3` <- diag(inv_metric$`3`)
  inv_metric$`4` <- diag(inv_metric$`4`)
}

fit <- mod$sample(data = stan_data, init = init2,
                   chains = 4, parallel_chains = 4,
                   iter_warmup = 0, iter_sampling = 1000,
                   seed = 123, adapt_engaged = FALSE,
                   inv_metric = inv_metric,
                   step_size = step_size)
# sampling phase causes no issue.

fit <- mod$sample(data = stan_data, init = init2,
                  chains = 4, parallel_chains = 4,
                  iter_warmup = 1000, iter_sampling = 1000,
                  seed = 123, adapt_engaged = TRUE,
                  inv_metric = inv_metric,
                  step_size = step_size)


#####################################################################
## Let's try ADVI

fit_vi <- mod$variational(data = stan_data, init = init)
fit_vi$summary()  # overestimates sigma

# ADVI won't work with the ela model because the ELBO cannot be
# evaluated for the initial variational distribution.

# fit_full$summary()
# fit_vi_full$summary()

