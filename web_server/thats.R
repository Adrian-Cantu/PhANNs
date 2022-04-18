library(tidyverse)
reticulate::use_condaenv(condaenv = 'tf', required = TRUE)

library(reticulate)
library(purrr)

#Rthats <- py$major_cap_thats
#Rnames <- py$major_cap_thats_n

Rthats_l <- list(py$major_cap_thats,py$minor_cap_thats,
                 py$baseplate_thats,py$major_tail_thats,
                 py$minor_tail_thats,py$portal_thats,
                 py$tail_fiber_thats,py$tail_shaft_thats,
                 py$collar_thats,py$htj_thats,
                 py$other_thats)

Rnames_l <- list(py$major_cap_thats_n,py$minor_cap_thats_n,
                 py$baseplate_thats_n,py$major_tail_thats_n,
                 py$minor_tail_thats_n,py$portal_thats_n,
                 py$tail_fiber_thats_n,py$tail_shaft_thats_n,
                 py$collar_thats_n,py$htj_thats_n,
                 py$other_thats_n)

P_classes <- c("Major_capsid","Minor_capsid","Baseplate",
                          "Major_tail","Minor_tail","Portal",
                          "Tail_fiber","Tail_shaft","Collar",
                          "HTJ","Other")

names(Rthats_l) <- P_classes
names(Rnames_l) <- P_classes

net_names <- c('net_01','net_02','net_03','net_04','net_05',
               'net_06','net_07','net_08','net_09','net_10')


kk <- imap(Rthats_l,function(results,names1){
  names(results) <- net_names
  prot_name <- as.vector(Rnames_l[[names1]])
  ret_list <- imap(results,function(x,names_net){
    
    colnames(x) <- P_classes
    rownames(x) <- prot_name
    ret <- x %>% as.data.frame() %>% 
      rownames_to_column(var='id') %>%
      pivot_longer(-id,names_to = 'score_class', values_to = 'score') %>%
      mutate(net_num = names_net) %>% 
      mutate(real_class=names1)
  })
  ret2 <- base::Reduce(rbind,ret_list)
  return(ret2)
})

final_result <- base::Reduce(rbind,kk)
write.table(final_result, file='PhANNs_test_10_nets.tsv', quote=FALSE, sep='\t', row.names = FALSE)
