#' generate mixture distribution of same family
#'
#' @param dist tfp distribution
#' @param nr_comps number of mixture components
#' @param trafos_each_param list of transformaiton applied before plugging
#' the linear predictor into the parameters of the distributions.
#' Should be of length #parameters of \code{dist}
#' @return returns function than can be used as argument \code{dist\_fun} for
#' @export
#'
mix_dist_maker <- function(
  dist = tfd_normal,
  nr_comps = 3,
  trafos_each_param = list(function(x) x, function(x) tf$add(1e-8, tfe(x)))
  ){

  stack <- function(x,ind=1:nr_comps) tf$stack(
    lapply(ind, function(j)
      tf_stride_cols(x,j)), 2L)

  mixdist = function(probs, params)
  {

    mix = tfd_categorical(probs = probs)
    this_components = do.call(dist, params)

    res_dist <- tfd_mixture_same_family(
      mixture_distribution=mix,
      components_distribution=this_components
    )

    return(res_dist)
  }

  trafo_fun <- function(x){

    c(probs = list(stack(x,1:nr_comps)),
      params = list(
        lapply(1:length(trafos_each_param),
               function(i)
                 stack(trafos_each_param[[i]](
                   tf_stride_cols(x,nr_comps +
                                    # first x for pis
                                    (i-1)*nr_comps +
                                    # then for each parameter
                                    (1:nr_comps))
                 )
                 )
        )
      )
    )

  }

  return(
    function(x) do.call(mixdist, trafo_fun(x))
  )

}
