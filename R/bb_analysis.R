library(tidyverse)

pres <- read_csv("output/validationbbprsent2F.csv")

table(pres$`breaking leaves`)[c(1, 2, 6)]
table(pres$`breaking leaves`)[c(1, 2, 6)] / sum(table(pres$`breaking leaves`)[c(1, 2, 6)])

abs <- read_tsv("output/validationbbabsent.txt")
