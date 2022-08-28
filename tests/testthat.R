library(testthat)
library(deepregression)
library(mixdistreg)

if (reticulate::py_module_available("tensorflow") & 
    reticulate::py_module_available("keras")){
  test_check("mixdistreg")
}