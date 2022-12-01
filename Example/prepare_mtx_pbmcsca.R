library(Seurat)
library(SeuratData)
data('pbmcsca')

library(dplyr)
library(useful)
library(Matrix)
library(reticulate)

use_python('/home/zfeng/miniconda3/envs/pyro/bin/python')
skl <- import('sklearn')


seurat <- NormalizeData(pbmcsca)
seurat <- FindVariableFeatures(seurat, nfeatures = 2000)
seurat <- ScaleData(seurat)

seurat <- RunPCA(seurat, verbose=FALSE)
seurat <- RunUMAP(seurat, dims = 1:30)


##########
cat('save dataset\n')

# save data
cells <- colnames(seurat)
hvgs <- VariableFeatures(seurat)


X <- GetAssayData(seurat, 'data')
X <- X[hvgs, cells] %>% as.matrix %>% t
spMat <- Matrix(X, sparse = TRUE)
cat(nrow(spMat), ',', ncol(spMat), '\n')
writeMM(spMat, file = 'pbmcsca.mtx')

# save cell
write.table(cells, file = 'pbmcsca_cell.txt', sep = '\n', row.names = F, col.names = F, quote = F)

# save gene
write.table(hvgs, file = 'pbmcsca_gene.txt', sep = '\n', row.names = F, col.names = F, quote = F)

# save label (text)

enc <- skl$preprocessing$OneHotEncoder(sparse=FALSE)$fit(seurat@meta.data[,c('CellType','Method'),drop=FALSE])
onehot <- enc$transform(seurat@meta.data[,c('CellType','Method'),drop=FALSE]) %>% as.data.frame
colnames(onehot) <- unlist(enc$categories_)
write.table(onehot, 
            file = 'pbmcsca_factors.txt', 
            sep = ',', row.names = F, col.names = T, quote = T)


