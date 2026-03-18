// Flexible Dirichlet-Multinomial (FDM) regression model
// Supports a single global mixing weight w_norm
// Source: EFDMReg package (Ascari & Migliorati, 2021)
// https://github.com/robertoascari/EFDMReg

data{
	int<lower=1> N;     // total number of observations
	int<lower=2> D;     // number of categories
	int<lower=0> n[N];  // number of trials per observation
	int<lower=1> K;     // number of predictors (including intercept)
	matrix[N,K] X;      // predictor design matrix
	int Y[N,D];         // response count matrix
	real sd_prior;      // prior standard deviation for beta coefficients
}

transformed data{
	matrix[N,D] Y_real;
	for(i in 1:N){
		Y_real[i,] = to_row_vector(Y[i,]);
	}
}

parameters {
	matrix[D-1,K] beta_raw;        // free regression coefficients
	real<lower=0> aplus;            // overall precision
	simplex[D] p;                   // baseline Dirichlet weights
	real<lower=0,upper=1> w_norm;   // single global mixing weight
}

transformed parameters{
	matrix[D,K] beta;
	matrix[N,D] mu;
	matrix[N,D] logits;

	for (l in 1:K) {
		beta[D,l] = 0.0;
	}
	for (k in 1:(D-1)) {
		for (l in 1:K) {
			beta[k,l] = beta_raw[k,l];
		}
	}

	for (i in 1:N) {
		for (m in 1:D){
			logits[i,m] = X[i,] * transpose(beta[m,]);
		}
		mu[i,] = to_row_vector(softmax(to_vector(logits[i,])));
	}
}

model {
	// Priors
	p ~ dirichlet(rep_vector(1, D));
	aplus ~ gamma(1*0.001, 0.001);
	w_norm ~ beta(30, 30);

	for (k in 1:(D-1)) {
		for (l in 1:K) {
			beta_raw[k,l] ~ normal(0, sd_prior);
		}
	}

	// Likelihood (Flexible Dirichlet-Multinomial)
	for (i in 1:N) {
		real temp;
		real temp2;
		vector[D] alpha;
		vector[D] Y_temp;
		vector[D] tau;
		temp  = 0.0;
		temp2 = 0.0;

		for(j in 1:D){
			temp2 = temp2 + p[j]*w_norm*fmin(1.0, mu[i,j]/p[j]);
		}

		for(j in 1:D){
			alpha[j] = aplus*((mu[i,j] - p[j]*w_norm*fmin(1.0, mu[i,j]/p[j]))/(1-temp2));
			tau[j]   = aplus*(w_norm*fmin(1.0, mu[i,j]/p[j])/(1-w_norm*fmin(1.0, mu[i,j]/p[j])));

			temp    = temp + lgamma(alpha[j]+Y[i,j]) - lgamma(alpha[j]) - lgamma(Y[i,j]+1);
			Y_temp[j] = log(p[j]) + lgamma(aplus+tau[j]) - lgamma(aplus+n[i]+tau[j])
			            + lgamma(alpha[j]) + lgamma(alpha[j]+tau[j]+Y[i,j])
			            - lgamma(alpha[j]+tau[j]) - lgamma(alpha[j]+Y[i,j]);
		}
		target += lgamma(n[i]+1) + temp + log_sum_exp(Y_temp);
	}
}

generated quantities {
	vector[N] log_lik;
	for (i in 1:N) {
		real temp;
		real temp2;
		vector[D] alpha;
		vector[D] Y_temp;
		vector[D] tau;
		temp  = 0.0;
		temp2 = 0.0;

		for(j in 1:D){
			temp2 = temp2 + p[j]*w_norm*fmin(1.0, mu[i,j]/p[j]);
		}

		for(j in 1:D){
			alpha[j] = aplus*((mu[i,j] - p[j]*w_norm*fmin(1.0, mu[i,j]/p[j]))/(1-temp2));
			tau[j]   = aplus*(w_norm*fmin(1.0, mu[i,j]/p[j])/(1-w_norm*fmin(1.0, mu[i,j]/p[j])));

			temp    = temp + lgamma(alpha[j]+Y[i,j]) - lgamma(alpha[j]) - lgamma(Y[i,j]+1);
			Y_temp[j] = log(p[j]) + lgamma(aplus+tau[j]) - lgamma(aplus+n[i]+tau[j])
			            + lgamma(alpha[j]) + lgamma(alpha[j]+tau[j]+Y[i,j])
			            - lgamma(alpha[j]+tau[j]) - lgamma(alpha[j]+Y[i,j]);
		}
		log_lik[i] = lgamma(n[i]+1) + temp + log_sum_exp(Y_temp);
	}
}
