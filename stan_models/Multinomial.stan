// Multinomial regression model
// Source: EFDMReg package (Ascari & Migliorati, 2021)
// https://github.com/robertoascari/EFDMReg

data{
	int<lower=1> N; // total number of observations
	int<lower=2> D; // number of categories
	int<lower=2> K; // number of predictor levels (including intercept)
	matrix[N,K] X; // predictor design matrix
	int Y[N,D];    // response count matrix
	real sd_prior; // prior standard deviation for beta coefficients
}

parameters {
	matrix[D-1,K] beta_raw; // free regression coefficients (last category constrained to 0)
}

transformed parameters{
	matrix[D,K] beta;   // full coefficient matrix
	matrix[N,D] mu;     // predicted category probabilities
	matrix[N,D] logits;

	for (l in 1:K) {
		beta[D,l] = 0.0;
	}
	for (k in 1:(D-1)) {
		for (l in 1:K) {
			beta[k,l] = beta_raw[k,l];
		}
	}

	for (n in 1:N) {
		for (m in 1:D){
			logits[n,m] = X[n,] * transpose(beta[m,]);
		}
		mu[n,] = to_row_vector(softmax(to_vector(logits[n,])));
	}
}

model {
	// Priors
	for (k in 1:(D-1)) {
		for (l in 1:K) {
			beta_raw[k,l] ~ normal(0, sd_prior);
		}
	}
	// Likelihood
	for (n in 1:N) {
		Y[n,] ~ multinomial(transpose(mu[n,]));
	}
}

generated quantities{
	vector[N] log_lik;
	for(n in 1:N){
		log_lik[n] = multinomial_lpmf(Y[n,] | transpose(mu[n,]));
	}
}
