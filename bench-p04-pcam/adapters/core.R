.libPaths(c("./.Rlibs", .libPaths()))
suppressPackageStartupMessages(library(jsonlite))
stored_X <<- NULL
model_params <<- NULL
R_op <<- NULL
pi_min <<- 0.1
pi_max <<- 10.0
geom_steps <<- 180
geometry_cache <<- list()
N <<- NULL

build_default_R <- function(N, gamma=0.2, delta=0.1, alpha=0.5, edge_p=0.1, seed=0) {
  set.seed(seed)
  A <- alpha * diag(N)
  upper <- matrix(runif(N*N) < edge_p, nrow=N)
  upper[lower.tri(upper, diag=TRUE)] <- 0
  adj <- upper + t(upper)
  
  deg <- rowSums(adj)
  deg_safe <- ifelse(deg > 0, deg, 1.0)
  d_inv_sqrt <- 1.0 / sqrt(deg_safe)
  
  normalised_adj <- sweep(sweep(adj, 1, d_inv_sqrt, "*"), 2, d_inv_sqrt, "*")
  L <- diag(N) - normalised_adj
  
  R <- A + gamma * L + delta * matrix(1, N, N)
  return(0.5 * (R + t(R)))
}

clip_and_normalise <- function(pi_vec, pi_min_val, pi_max_val) {
  pi_vec <- as.numeric(pi_vec)
  if(any(!is.finite(pi_vec))) return(rep(1, N))
  
  for(i in 1:20) {
    pi_vec <- pmin(pmax(pi_vec, pi_min_val), pi_max_val)
    m <- mean(pi_vec)
    if(m <= 1e-12) return(rep(1, N))
    pi_vec <- pi_vec / m
    
    within_bounds <- (min(pi_vec) >= pi_min_val - 1e-9) && (max(pi_vec) <= pi_max_val + 1e-9)
    mean_ok <- abs(mean(pi_vec) - 1.0) < 1e-8
    if(within_bounds && mean_ok) break
  }
  return(pmin(pmax(pi_vec, pi_min_val), pi_max_val))
}

softmax_func <- function(a) {
  z <- model_params$beta * (stored_X %*% a)
  z <- z - max(z)
  e <- exp(z)
  return(e / sum(e))
}

gradient <- function(a) {
  s <- softmax_func(a)
  return(R_op %*% a - model_params$eta * (t(stored_X) %*% s))
}

hessian <- function(a) {
  s <- softmax_func(a)
  s_vec <- as.vector(s)
  D <- diag(s_vec) - outer(s_vec, s_vec)
  H <- R_op - model_params$eta * model_params$beta * (t(stored_X) %*% D %*% stored_X)
  return(0.5 * (H + t(H)))
}

run_dynamics <- function(a0, pi_vec, u_const=NULL, T_max_val=NULL) {
  pi_vec <- clip_and_normalise(pi_vec, model_params$pi_min, model_params$pi_max)
  a <- as.vector(a0)
  T_val <- if(is.null(T_max_val)) model_params$T_max else T_max_val
  T_in <- model_params$T_in
  dt <- model_params$dt
  
  for(t in 0:(T_val-1)) {
    g <- as.vector(gradient(a))
    update <- -pi_vec * g
    if(!is.null(u_const) && t < T_in) {
      update <- update + u_const
    }
    a_new <- a + dt * update
    if(sqrt(sum((a_new - a)^2)) < model_params$tol) {
      a <- a_new
      break
    }
    a <- a_new
  }
  return(a)
}

find_equilibrium <- function(x0) {
  return(run_dynamics(x0, rep(1, N), u_const=NULL))
}

# --- Agent specific logic ---

EPS <- 1e-12
AMPLITUDE_EPS <- 0.2
NEAR_PATTERN_COSINE <- 0.84
MARGIN_COSINE <- 0.76
MARGIN_GAP <- 0.28

project_pi <- function(pi_vec) {
  pi_vec <- as.numeric(pi_vec)
  if(any(!is.finite(pi_vec))) return(rep(1, N))
  
  for(i in 1:20) {
    pi_vec <- pmin(pmax(pi_vec, pi_min), pi_max)
    m <- mean(pi_vec)
    if(m <= EPS) return(rep(1, N))
    pi_vec <- pi_vec / m
  }
  return(pmin(pmax(pi_vec, pi_min), pi_max))
}

condition_and_grad <- function(H, y) {
  pi_vec <- project_pi(exp(y - mean(y)))
  d <- sqrt(pmax(pi_vec, EPS))
  S <- sweep(sweep(H, 1, d, "*"), 2, d, "*")
  S <- 0.5 * (S + t(S))
  
  e <- eigen(S, symmetric=TRUE)
  eigs <- e$values
  vecs <- e$vectors
  
  valid_idx <- which(eigs > 1e-9)
  if(length(valid_idx) < 2) {
    return(list(obj=Inf, grad=rep(0, N), pi=pi_vec))
  }
  
  hi <- valid_idx[1]
  lo <- valid_idx[length(valid_idx)]
  
  obj <- log(eigs[hi]) - log(eigs[lo])
  grad <- vecs[, hi]^2 - vecs[, lo]^2
  grad <- grad - mean(grad)
  return(list(obj=obj, grad=grad, pi=pi_vec))
}

optimise_diagonal <- function(H) {
  H <- 0.5 * (H + t(H))
  diag_H <- pmax(diag(H), 1e-8)
  y <- log(project_pi(1.0 / diag_H))
  
  res <- condition_and_grad(H, y)
  best_obj <- res$obj
  best_pi <- res$pi
  lr <- 2.0
  
  for(i in 1:geom_steps) {
    res <- condition_and_grad(H, y)
    obj <- res$obj
    grad <- res$grad
    
    candidate_y <- y - lr * grad
    cand_res <- condition_and_grad(H, candidate_y)
    candidate_obj <- cand_res$obj
    candidate_pi <- cand_res$pi
    
    if(candidate_obj < obj) {
      y <- candidate_y
      lr <- lr * 1.02
      if(candidate_obj < best_obj) {
        best_obj <- candidate_obj
        best_pi <- candidate_pi
      }
    } else {
      lr <- lr * 0.5
      if(lr < 1e-5) lr <- 0.1
    }
  }
  return(best_pi)
}

geometry_precision <- function(idx) {
  idx_char <- as.character(idx)
  if(!is.null(geometry_cache[[idx_char]])) {
    return(geometry_cache[[idx_char]])
  }
  
  eq <- find_equilibrium(stored_X[idx, ])
  H <- hessian(eq)
  pi_vec <- optimise_diagonal(H)
  geometry_cache[[idx_char]] <<- pi_vec
  return(pi_vec)
}

near_stored_pattern <- function(sims) {
  if(length(sims) == 1) return(sims[1] > NEAR_PATTERN_COSINE)
  
  sorted_sims <- sort(sims, decreasing=TRUE)
  best <- sorted_sims[1]
  margin <- sorted_sims[1] - sorted_sims[2]
  
  return(best > NEAR_PATTERN_COSINE || (best > MARGIN_COSINE && margin > MARGIN_GAP))
}

nearest_pattern <- function(q) {
  n <- sqrt(sum(q^2))
  if(n <= EPS) return(list(idx=NULL, sims=NULL))
  
  sims <- as.vector(stored_X %*% (q / n))
  return(list(idx=which.max(sims), sims=sims))
}

retrieval_precision <- function(q) {
  return(project_pi(1.0 / (abs(q) + AMPLITUDE_EPS)))
}

predict_precision <- function(q) {
  q <- as.numeric(q)
  res <- nearest_pattern(q)
  if(is.null(res$idx)) {
    return(rep(1, N))
  }
  
  if(near_stored_pattern(res$sims)) {
    return(geometry_precision(res$idx))
  }
  
  return(retrieval_precision(q))
}

# --- IO Loop ---

init_agent <- function(data) {
  stored_X <<- as.matrix(data$stored_patterns)
  model_params <<- data$model_params
  N <<- ncol(stored_X)
  
  pi_min <<- if(!is.null(model_params$pi_min)) model_params$pi_min else 0.1
  pi_max <<- if(!is.null(model_params$pi_max)) model_params$pi_max else 10.0
  geom_steps <<- if(N <= 96) 300 else 180
  geometry_cache <<- list()
  
  if(!is.null(model_params$R)) {
    R_op <<- as.matrix(model_params$R)
  } else {
    R_op <<- build_default_R(N)
  }
}

main_loop <- function() {
  con <- file("stdin", open="r")
  while(TRUE) {
    line <- readLines(con, n=1, warn=FALSE)
    if(length(line) == 0) break
    if(trimws(line) == "") next
    
    req <- tryCatch({
      fromJSON(line, simplifyVector=TRUE, simplifyMatrix=TRUE)
    }, error = function(e) NULL)
    
    if(is.null(req)) {
      cat(toJSON(list(error="Invalid JSON"), auto_unbox=TRUE), "\n")
      next
    }
    
    if(req$type == "init") {
      init_agent(req$data)
      cat(toJSON(list(status="ok"), auto_unbox=TRUE), "\n")
    } else if(req$type == "predict") {
      pi_out <- predict_precision(req$q)
      cat(toJSON(list(pi=pi_out), digits=8), "\n")
    } else if(req$type == "exit") {
      break
    }
  }
  close(con)
}

if(!interactive()) {
  main_loop()
}
