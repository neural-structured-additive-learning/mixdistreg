context("Helper Functions and Methods")

check_funs <- function(trafo_fun, sizes, mixdist, type=c("general", "same"))
{
  
  type = match.arg(type)
  
  total_cols <- length(sizes) + sum(sizes)
  n_rows <- 4
  
  # try out with tensors
  x <- tf$constant(matrix(1:(n_rows*total_cols), ncol=total_cols
                          # 2 for mixture
                          # 2 for normal
                          # 1 for t-dist
  ), dtype="float32")
  
  transformed_x <- trafo_fun(x)
  
  expect_is(transformed_x, "list")
  expect_length(transformed_x, 2L)
  expect_equal(
    names(transformed_x),
    c("probs", "params")
  )
  expect_equal(ncol(as.array(transformed_x$probs)[,1,]), length(sizes))
  expect_equal(ncol(as.matrix(transformed_x$params)), sum(sizes))
  # check probabilities
  expect_true(all.equal(
    round(rowSums(as.array(transformed_x$probs)[,1,]), 5),
    rep(1, n_rows)))
  
  tc <- switch(type,
               general = "tensorflow_probability.python.distributions.mixture.Mixture",
               same = "tensorflow_probability.python.distributions.mixture_same_family.MixtureSameFamily")
  
  # check dist
  expect_true(inherits(do.call(mixdist, transformed_x),
                       tc))
  
}

test_that("gen_mix_dist_maker", {

  res <- gen_mix_dist_maker(
    families = c("normal", "normal"),
    nr_distributions = 2L
  )
  # get functions in res
  mixdist <- get("mixdist", environment(res))
  trafo_fun <- get("trafo_fun", environment(res))
  # check
  check_funs(trafo_fun, sizes=c(2,2), mixdist)
  
  res <- gen_mix_dist_maker(
    families = c("normal", "student_t"),
    nr_distributions = 2L
  )
  # get functions in res
  mixdist <- get("mixdist", environment(res))
  trafo_fun <- get("trafo_fun", environment(res))
  # check
  check_funs(trafo_fun, sizes=c(2,1), mixdist)
  
  res <- gen_mix_dist_maker(
    families = c("normal", "student_t", "gumbel"),
    nr_distributions = 3L
  )
  # get functions in res
  mixdist <- get("mixdist", environment(res))
  trafo_fun <- get("trafo_fun", environment(res))
  # check
  check_funs(trafo_fun, sizes=c(2,1,2), mixdist)
  
  res <- gen_mix_dist_maker(
    families = c("normal", "student_t", "student_t_ls"),
    nr_distributions = 3L
  )
  # get functions in res
  mixdist <- get("mixdist", environment(res))
  trafo_fun <- get("trafo_fun", environment(res))
  # check
  check_funs(trafo_fun, sizes=c(2,1,3), mixdist)
  
  res <- gen_mix_dist_maker(
    families = c("normal"),
    nr_distributions = 3L
  )
  # get functions in res
  mixdist <- get("mixdist", environment(res))
  trafo_fun <- get("trafo_fun", environment(res))
  # check
  check_funs(trafo_fun, sizes=c(2,1,3), mixdist, "same")
  
  
})

test_that("methods", {
  
  set.seed(32)
  n <- 1000
  data = data.frame(matrix(rnorm(4*n), c(n,4)))
  colnames(data) <- c("x1","x2","x3","xa")
  formula <- ~ 1 + deep_model(x1,x2,x3) + s(xa) + x1
  
  deep_model <- function(x) x %>%
    layer_dense(units = 32, activation = "relu", 
                use_bias = FALSE) %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 8, activation = "relu") %>%
    layer_dense(units = 1, activation = "linear")
  
  y <- rnorm(n) + data$xa^2 + data$x1
  
  minmod <- mixdistreg(
    y,
    families = c("student_t_ls", "gumbel", "student_t"),
    nr_comps = 3L,
    list_of_formulas = list(~1 + s(xa), ~1, ~1 + x1,
                            ~1 + x2, ~1 + xa,
                            ~1 + x3),
    formula_mixture = ~1 + x1 + x2,
    data = data  
  ) 
  
  cf1 <- coef(minmod, which_dist = 1)
  expect_equal(names(cf1), c("x1", "x2", "(Intercept)"))
  
  cf1 <- coef(minmod, which_dist = "mixture")
  expect_equal(names(cf1), c("x1", "x2", "(Intercept)"))
  
  cf2 <- coef(minmod, which_dist = 2)
  expect_equal(names(cf2), c("s(xa)", "(Intercept)"))
  
  cf2 <- coef(minmod, which_dist = "student_t_ls")
  expect_equal(names(cf2), c("s(xa)", "(Intercept)"))
  
  cf3 <- coef(minmod, which_dist = 3, which_param = 2)
  expect_equal(names(cf3), c("xa", "(Intercept)"))
  
  cf3 <- coef(minmod, which_dist = "gumbel", which_param = 2)
  expect_equal(names(cf3), c("xa", "(Intercept)"))
  
  cf4 <- coef(minmod, which_dist = 4)
  expect_equal(names(cf4), c("x3", "(Intercept)"))
  
  cf4 <- coef(minmod, which_dist = "student_t")
  expect_equal(names(cf4), c("x3", "(Intercept)"))

  # test combination of dist and nr
  minmod <- mixdistreg(
    y,
    families = c("normal", "normal", "student_t"),
    nr_comps = 3L,
    list_of_formulas = list(~1 + s(xa), ~1 + x1,
                            ~1 + x2, ~1 + xa,
                            ~1 + x3),
    formula_mixture = ~1 + x1 + x2,
    data = data  
  ) 
  
  cf1 <- coef(minmod, which_dist = "normal1")
  expect_equal(names(cf1), c("s(xa)", "(Intercept)"))

  cf2 <- coef(minmod, which_dist = "normal2")
  expect_equal(names(cf2), c("x2", "(Intercept)"))
  
})