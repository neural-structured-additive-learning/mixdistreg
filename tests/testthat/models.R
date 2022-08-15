context("Model Functions")

# -------------------- data generation ----------------------- #
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

# ------------------ test functions ------------------------- #

check_methods <- function(m, newdata, test_plots = TRUE)
{
  # fit
  hist <- m %>% fit(epochs = 2, verbose = FALSE)
  expect_is(hist, "keras_training_history")
  expect_true(all(!is.nan(hist$metrics$loss)))
  expect_true(all(!is.nan(hist$metrics$val_loss)))
  
  # plot
  if (test_plots) {
    pm <- plot(m)
    expect_is(pm, "list")
  }
  
  # coef
  ch1 <- coef(m)
  expect_is(ch1, "list")
  ch2 <- coef(m, which_param = 2)
  expect_is(ch2, "list")
  
  # fitted
  fitt <- m %>% fitted()
  expect_is(fitt, "matrix")
  
  # predict
  preds <- predict(m, newdata = newdata)
  expect_is(preds, "matrix")
  
  # distribution
  dist <- get_distribution(m, data= newdata)
  expect_true(inherits(dist, "tensorflow_probability.python.distributions.distribution.Distribution"))
  
}


test_that("mixdistreg", {

  # Minimal specifications: No predictor case 2
  minmod <- mixdistreg(
    y,
    type = c("same"),
    nr_comps = 2L,
    list_of_formulas = list(~1, ~1,
                            ~1, ~1),
    formula_mixture = ~1,
    data = data  
    )
  check_methods(minmod, data, FALSE)
  
  # Minimal specifications: No predictor case 3
  minmod <- mixdistreg(
    y,
    type = c("same"),
    nr_comps = 3L,
    list_of_formulas = list(~1, ~1,
                            ~1, ~1,
                            ~1, ~1),
    formula_mixture = ~1,
    data = data  
  )
  check_methods(minmod, data, FALSE)
  
  # Minimal specifications: No predictor case general
  minmod <- mixdistreg(
    y,
    type = c("general"),
    families = c("normal", "normal", "normal"),
    nr_comps = 3L,
    list_of_formulas = list(~1, ~1,
                            ~1, ~1,
                            ~1, ~1),
    formula_mixture = ~1,
    data = data  
  )
  check_methods(minmod, data, FALSE) 
  
  # Minimal specifications: Different distributions
  minmod <- mixdistreg(
    y,
    type = c("general"),
    families = c("student_t_ls", "gumbel", "student_t"),
    nr_comps = 3L,
    list_of_formulas = list(~1, ~1, ~1,
                            ~1, ~1,
                            ~1),
    formula_mixture = ~1,
    data = data  
  )
  check_methods(minmod, data, FALSE) 
  
  # Features in mixture
  minmod <- mixdistreg(
    y,
    type = c("general"),
    families = c("student_t_ls", "gumbel", "student_t"),
    nr_comps = 3L,
    list_of_formulas = list(~1, ~1, ~1,
                            ~1, ~1,
                            ~1),
    formula_mixture = ~1 + x1,
    data = data  
  )
  check_methods(minmod, data, FALSE) 
  
  # Features in predictors
  minmod <- mixdistreg(
    y,
    type = c("general"),
    families = c("student_t_ls", "gumbel", "student_t"),
    nr_comps = 3L,
    list_of_formulas = list(~1 + s(xa), ~1, ~1 + x1,
                            ~1 + x2, ~1,
                            ~1 + x3),
    formula_mixture = ~1,
    data = data  
  )
  check_methods(minmod, data, FALSE) 
  
  # Features in predictors and mixture
  minmod <- mixdistreg(
    y,
    type = c("general"),
    families = c("student_t_ls", "gumbel", "student_t"),
    nr_comps = 3L,
    list_of_formulas = list(~1 + s(xa), ~1, ~1 + x1,
                            ~1 + x2, ~1,
                            ~1 + x3),
    formula_mixture = ~1 + x2,
    data = data  
  )
  check_methods(minmod, data, FALSE) 
  
  # Deep model
  minmod <- mixdistreg(
    y,
    type = c("same"),
    nr_comps = 2L,
    list_of_formulas = list(formula, ~1,
                            ~1, ~1),
    formula_mixture = ~1,
    data = data,
    list_of_deep_models = list(deep_model = deep_model)
  )
  check_methods(minmod, data, FALSE)
  
  # Deep model in mixture
  minmod <- mixdistreg(
    y,
    type = c("same"),
    nr_comps = 2L,
    list_of_formulas = list(~1, ~1,
                            ~1, ~1),
    formula_mixture = formula,
    data = data,
    list_of_deep_models = list(deep_model = deep_model)
  )
  check_methods(minmod, data, FALSE)
  
  # Different trafos
  
  # Deep model
  minmod <- mixdistreg(
    y,
    type = c("same"),
    nr_comps = 2L,
    list_of_formulas = list(formula, ~1,
                            ~1, ~1),
    formula_mixture = ~1,
    data = data,
    list_of_deep_models = list(deep_model = deep_model),
    trafos_each_param = list(list(
      function(x) x, function(x) tf$nn$softplus(x)
    ),
    list(
      function(x) x, function(x) tf$nn$softplus(x)
    ))
  )
  check_methods(minmod, data, FALSE)
  
})

test_that("sammer", {

  # check that the wrapper works as designed
  # 
  # first create model via wrapper
  modsame <- sammer(
    y,
    family = "gumbel",
    nr_comps = 2L,
    list_of_formulas = list(~1, ~1 + x1),
    formula_mixture = ~1,
    data = data,
  )
  # now manually
  modman <- mixdistreg(
    y,
    families = "gumbel",
    type = c("same"),
    nr_comps = 2L,
    list_of_formulas = list(~1, ~1 + x1)[rep(1:2, 2)],
    formula_mixture = ~1,
    data = data
  )
  
  expect_true(
    all.equal(modsame$init_params$mixture_specification,
              modman$init_params$mixture_specification)
  )

})

test_that("inflareg", {

  # one inflation value
  modinf <- inflareg(
    y,
    family = "normal",
    list_of_formulas = list(~1, ~1),
    formula_inflation = ~1,
    list_of_deep_models = NULL,
    inflation_values = 3,
    data = data,
  )
  # check if families are correct
  expect_equal(
    modinf$init_params$mixture_specification$families,
    c("normal", "deterministic")
  )
  # check methods
  check_methods(modinf, data, FALSE)
  
  # three inflation value
  modinf <- inflareg(
    y,
    family = "normal",
    list_of_formulas = list(~1, ~1),
    formula_inflation = ~1,
    list_of_deep_models = NULL,
    inflation_values = 1:3,
    data = data,
  )
  # check if families are correct
  expect_equal(
    modinf$init_params$mixture_specification$families,
    c("normal", "deterministic", "deterministic", "deterministic")
  )
  # check methods
  check_methods(modinf, data, FALSE)

})

test_that("Inflation wrapper", {

  # check that the wrapper works as designed
  
  # zin
  modwrap <- zinreg(
    y,
    family = "normal",
    list_of_formulas = list(~1, ~1),
    formula_inflation = ~1,
    list_of_deep_models = NULL,
    data = data,
  )
  
  modman <- inflareg(
    y,
    family = "normal",
    list_of_formulas = list(~1, ~1),
    formula_inflation = ~1,
    list_of_deep_models = NULL,
    inflation_values = 0,
    data = data,
  )
  
  expect_equal(
    modman$init_params$mixture_specification$inflation_values,
    modwrap$init_params$mixture_specification$inflation_values,
  )
  
  # oin
  modwrap <- oinreg(
    y,
    family = "normal",
    list_of_formulas = list(~1, ~1),
    formula_inflation = ~1,
    list_of_deep_models = NULL,
    data = data,
  )
  
  modman <- inflareg(
    y,
    family = "normal",
    list_of_formulas = list(~1, ~1),
    formula_inflation = ~1,
    list_of_deep_models = NULL,
    inflation_values = 1,
    data = data,
  )
  
  expect_equal(
    modman$init_params$mixture_specification$inflation_values,
    modwrap$init_params$mixture_specification$inflation_values,
  )
  
  # zoin
  modwrap <- zoinreg(
    y,
    family = "normal",
    list_of_formulas = list(~1, ~1),
    formula_inflation = ~1,
    list_of_deep_models = NULL,
    data = data,
  )
  
  modman <- inflareg(
    y,
    family = "normal",
    list_of_formulas = list(~1, ~1),
    formula_inflation = ~1,
    list_of_deep_models = NULL,
    inflation_values = 0:1,
    data = data,
  )
  
  expect_equal(
    modman$init_params$mixture_specification$inflation_values,
    modwrap$init_params$mixture_specification$inflation_values,
  )

})
