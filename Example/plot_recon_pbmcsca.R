library(Seurat)
library(SeuratData)
data('pbmcsca')

library(data.table)
library(useful)
library(dplyr)
library(ggplot2)
library(cowplot)

data <- fread('pbmcsca_mutate_express.txt', data.table=FALSE, nThread = 20)
rownames(data) <- data[,1]
data <- data[,-1]

data <- data %>% as.matrix %>% t
data <- 10 * (data/max(data))


########################################
recon <- CreateSeuratObject(data, meta.data = pbmcsca@meta.data[colnames(data), ])
recon <- SetAssayData(recon, slot = 'data', as.matrix(data))
recon <- FindVariableFeatures(recon, nfeatures=2000)
recon <- ScaleData(recon)

recon <- RunPCA(recon, verbose=FALSE)
recon <- RunUMAP(recon, dims=1:50, metric='euclidean')

recon <- FindNeighbors(recon, reduction = 'pca', dims = 1:30, force.recalc = TRUE)
recon <- FindClusters(recon, resolution = 0.02)

skl$metrics$adjusted_rand_score(recon$CellType, recon$seurat_clusters)
skl$metrics$adjusted_mutual_info_score(recon$CellType, recon$seurat_clusters)


DimPlot(recon, group.by = 'Method')
DimPlot(recon, group.by = 'CellType')


pos <- Embeddings(recon, reduction = 'pca')
write.csv(pos, file = 'pbmcsca_scc_pca.csv')

pos <- Embeddings(recon, reduction = 'umap')
write.csv(pos, file = 'pbmcsca_scc_umap.csv')
