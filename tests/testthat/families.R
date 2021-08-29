context("main")

test_that("mixdists", {
  mxdist = mix_dist_maker()
  expect_is(mxdist, "function")
  mkd = mxdist(matrix(rep(0.33, 12), ncol=12))
  expect_is(mkd, "python.builtin.object")
  expect_is(mkd$cdf, "python.builtin.method")
  expect_true(as.numeric(mkd$log_prob(1)) < 0)
})
