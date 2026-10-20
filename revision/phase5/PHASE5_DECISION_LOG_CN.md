# Phase 5: 审稿意见与实验状态清单

## Reviewer 2 Major 7（精度匹配的标量比较）

已写代码、已做 CPU 独立测试：

1. 固定 N 项 Lagrange--Bürmann series（含 p 接近 1 的困难区间）。
2. 后验 normalized-residual 检查、需要时 bisection fallback 的实际可用基线（它**不是**原 Mellin--Barnes hybrid）。
3. 在**同一批新输入**上复用 Phase 3 现有 Chebyshev / projected-Newton / bisection 实现。
4. 对每个 method/configuration 分别记录 absolute/normalized error、stationarity residual、objective gap、support mismatch、raw online time，单独记录 offline preparation。
5. 单独提供文献 [24] 原 Matlab MB 例程的近阈值输入与运行桥接（**尚未执行 MATLAB**）。

下一阶段：在用户 VS Code 的**真实工程**内运行 `--existing-methods all`，确认 API 和基线复现。再在 A100 同一输入生成器上做 pilot 和全 p grid。过去 Phase 1 的 CSV 不得与 Phase 5 的新输入/计时混成一份“同条件测量”。

尚未完成：MATLAB 原 MB 源码的独立原位精度/成本对照；严格 interval-certified 浮点证书；Phase5 的正式 GPU 测量与归档。

## Reviewer 2 Major 6（inexact prox safeguard）

已写代码并通过 CPU 测试：可选择 `--guard-chebyshev`，将计算得出的非零候选与零点的 proximal objective 进行 float64 比较，未可靠胜出则回零。该选项仅用于**新的 Phase5 标量 benchmark**，不影响历史 Phase4 A100 Lasso 数值。

尚未完成：把旧的 `Reviewer2_Objective_Safeguard_Code_Patch.zip` 与当前 Phase 3 源码集成、重新运行 GPU 计时、验证整条 FB 的模型下降条件和残差条件。仅比较“candidate vs zero”不能推出全局 PG convergence。

对于 F(x)=h(x)+g(x)，要引入非凸 inexact-prox convergence 所需的两类证书，例如

    Psi_k(x_{k+1}) <= Psi_k(x_k) + delta_k,
    || r_{k+1} || <= eta_k,

并说明 `sum_k delta_k < infinity` 与 `eta_k -> 0`，还需要明确计算中非零坐标上的选择与精度。现有固定度 Chebyshev 错误非消失，不保证以上条件自动成立。

## 目前不需要重跑的项目

Phase 1 的 18 个三 backend accuracy matched 结果、Phase2 阈值测试、Phase4 的完整 6 case / 54 method-seed A100 Lasso 数据均已归档。本 Phase 5 将产生 **新的** scalar benchmark，而不是重复 / 覆盖这些旧数据。
