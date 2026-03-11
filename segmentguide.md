

你需要同时满足的硬约束（建议全程当作 hard constraints）：

1. 单条字幕时长：最短 5/6 秒、最长 7 秒。 ([Netflix 合作伙伴帮助中心][1])
2. 单条字幕必须 1 行（你额外要求）；Netflix 也明确“通常保持一行，除非超过字符限制”。 ([Netflix 合作伙伴帮助中心][1])
3. 字符数：英文每行 42 characters。 ([Netflix 合作伙伴帮助中心][2])
4. 读速：成人节目最高 20 characters/sec。 ([Netflix 合作伙伴帮助中心][2])
5. 文本规范：不允许双空格；不允许 en/em dash（避免输出里出现长破折号）。 ([Netflix 合作伙伴帮助中心][2])

下面给一套“效果稳定、允许慢但要稳”的分段方法（把行内换行规则直接并入句子级切分）。

一、先把“无标点口语”变成“可断句文本”
用标点与大小写恢复模型先处理 ASR 文本，然后再切分。NeMo 的 Punctuation & Capitalization 模型就是为“让 ASR 输出更可读”设计的：逐词预测后继标点，并预测是否需要大写。 ([NVIDIA Docs][3])
你不用追求它 100% 准，只要它能提供“句末/逗号/疑问”的强提示，后面 DP 才能切得更像人工。

二、生成候选切点：只在“可读”位置允许切（把换行规则当切分规则）
Netflix 的 Line Treatment 里给了很明确的“建议断开位置”和“禁止断开结构”。 ([Netflix 合作伙伴帮助中心][1])
因为你不允许两行，我建议直接把它们转成字幕边界规则：

A. 作为“优先切点”（加奖励）

1. 标点后（尤其句号/问号/感叹号；逗号次之）。 ([Netflix 合作伙伴帮助中心][1])
2. 在连词前切（before conjunctions）。 ([Netflix 合作伙伴帮助中心][1])
3. 在介词前切（before prepositions）。 ([Netflix 合作伙伴帮助中心][1])
4. 词间停顿（用时间戳）：pause ≥0.8s 视为强切点；0.3–0.8s 视为弱切点（vlog 断句随意时，停顿通常比语法更可靠）。

B. 作为“禁止/强惩罚切点”（避免把一个结构撕开）
Netflix 明确不应拆开这些结构（原本是“不要在两行之间拆”，你现在把它当成“不要跨字幕拆”）： ([Netflix 合作伙伴帮助中心][1])

* 冠词/限定词 + 名词（a/the/this/my … + noun）
* 形容词 + 名词
* 名 + 姓（可用大写模式或 NER 检测）
* 主语代词 + 动词（I/you/he… 紧跟动词/助动）
* 介词性动词/短语动词 + 其介词/小品词（look at / give up 这类）
* 助动词/反身代词/否定词 + 动词（don’t + go, can + see, won’t + do 等）

实现上你不必做全依存句法也能有不错效果：用词表 + 正则（限定词、主语代词、否定词、常见介词、常见 phrasal particles）即可；如果你不在意耗时，再加一层 spaCy dependency parse，把“det→noun、amod→noun、aux/neg→verb、compound name”等关系作为强惩罚，会更稳。

三、全局最优切分：用动态规划把硬约束一次性压住
逐条“遇到停顿就切/遇到逗号就切”在 vlog 上很容易碎；建议做 DP（最短路）全局优化。

1. 输入：词序列 w1…wn，每个词有 (start_i, end_i)；文本已经过标点恢复与清洗（去双空格等）。 ([Netflix 合作伙伴帮助中心][2])
2. 枚举候选字幕片段 i→j（从第 i 个词到第 j 个词），先过硬约束过滤：

* duration = end_j − start_i 必须在 [0.833…, 7.0] 秒内。 ([Netflix 合作伙伴帮助中心][1])
* 渲染成 1 行字符串后，len_chars ≤ 42。 ([Netflix 合作伙伴帮助中心][2])
* len_chars / duration ≤ 20（成人读速）。 ([Netflix 合作伙伴帮助中心][2])
  不满足直接丢弃（这是你“只要一行且 Netflix 严格”的关键）。

3. 对通过过滤的候选片段计算 cost（越小越好），推荐这些项：

* 结尾奖励：以 . ? ! 结尾强奖励；以 , 结尾弱奖励；无标点不奖。
* 停顿奖励：片段末尾到下一个词的 pause 越大越奖励（例如 pause≥0.8s 强奖）。
* 结构撕裂惩罚：如果片段边界落在上面“禁止拆分结构”中，强惩罚。 ([Netflix 合作伙伴帮助中心][1])
* 读速贴顶惩罚：当 CPS 接近 20 时增加惩罚（逼算法宁可更频繁切、或触发删减）。 ([Netflix 合作伙伴帮助中心][2])
* 时长形状惩罚（可选）：轻微惩罚太贴近 0.833s（防“闪字幕”），但不要违反硬约束。Netflix 的 timing 指南也强调要避免“flashy sections”，需要全片回看并修正。 ([Netflix 合作伙伴帮助中心][4])

4. DP：dp[j] = min_i (dp[i-1] + cost(i→j))，并回溯得到最优切分序列。

四、vlog 口语“随意断句”的特别处理：以“删减/压缩”作为合规手段
vlog 常见 filler（um/uh/like/you know/I mean）会把 CPS 和 42 字符推爆。Netflix 英语指南明确：为满足 reading speed，优先用“reduction/deletion/condensing”；并且像 “hmm/um/whoa …” 这类拟声/语气词可以在需要时为读速或空间目的而减少。 ([Netflix 合作伙伴帮助中心][2])
所以建议你在 DP 之前或 DP 过程中加入一个“可选删减层”：

* 仅当某段无法同时满足 42 字符与 20 CPS 时，允许删掉低信息 filler（优先删 “you know / like / kind of / sort of / I mean / basically”等），再重新计算候选段是否可行。
* 不要做语义改写（尤其你要严格风格时）；删减要可解释、可回滚。

五、输出 SRT 前的硬质检（建议自动化）

* 严查：所有字幕 1 行、≤42 字符、≤20 CPS、时长在 [5/6, 7] 秒、无重叠、无双空格。 ([Netflix 合作伙伴帮助中心][1])
* 全片回看抓“闪字幕/切得太碎/语义被撕裂”的段落（Netflix timing 指南明确要求看回放并修正 flashy 段）。 ([Netflix 合作伙伴帮助中心][4])


[1]: https://partnerhelp.netflixstudios.com/hc/en-us/articles/215758617-Timed-Text-Style-Guide-General-Requirements "Timed Text Style Guide: General Requirements – Netflix | Partner Help Center"
[2]: https://partnerhelp.netflixstudios.com/hc/en-us/articles/217350977-English-USA-Timed-Text-Style-Guide "English (USA) Timed Text Style Guide – Netflix | Partner Help Center"
[3]: https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/nlp/punctuation_and_capitalization.html?utm_source=chatgpt.com "Punctuation and Capitalization Model - NVIDIA Documentation Hub"
[4]: https://partnerhelp.netflixstudios.com/hc/en-us/articles/360051554394-Timed-Text-Style-Guide-Subtitle-Timing-Guidelines?utm_source=chatgpt.com "Timed Text Style Guide: Subtitle Timing Guidelines"
