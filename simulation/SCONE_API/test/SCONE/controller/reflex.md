# 1. 股四頭肌的反射公式 (站立期)

$
S_{\text{VAS}}(t) = S_{0,\text{VAS}} + G_{\text{VAS}} F_{\text{VAS}}(t - \Delta t_{\text{VAS}}) - k_{\phi} \Delta \phi_k(t - \Delta t_k)
$

- \(S_{\text{VAS}}(t)\)：股四頭肌的刺激信號
- \(S_{0,\text{VAS}}\)：預先設定的刺激水平
- \(G_{\text{VAS}}\)：正向力反饋的增益
- \(F_{\text{VAS}}(t - \Delta t_{\text{VAS}})\)：來自股四頭肌的力反饋，經過時間延遲 \(\Delta t_{\text{VAS}}\)
- \(k_{\phi}\)：膝關節角度反饋的比例增益
- \(\Delta \phi_k\)：膝關節的角度偏差
- \(t - \Delta t_k\)：膝蓋角度變化的延遲時間

# 2. 比目魚肌的反射公式 (站立期)

$
S_{\text{SOL}}(t) = S_{0,\text{SOL}} + G_{\text{SOL}} F_{\text{SOL}}(t - \Delta t_{\text{SOL}})
$

- \(S_{\text{SOL}}(t)\)：比目魚肌的刺激信號
- \(S_{0,\text{SOL}}\)：預先設定的刺激水平
- \(G_{\text{SOL}}\)：正向力反饋的增益
- \(F_{\text{SOL}}(t - \Delta t_{\text{SOL}})\)：來自比目魚肌的力反饋，經過時間延遲 \(\Delta t_{\text{SOL}}\)

# 3. 脛前肌的反射公式 (站立期)

$
S_{\text{TA}}(t) = S_{0,\text{TA}} + G_{\text{TA}} (\ell_{\text{CE,TA}} - \ell_{\text{off,TA}})(t - \Delta t_{\text{TA}}) - G_{\text{SOLTA}} F_{\text{SOL}}(t - \Delta t_{\text{SOL}})
$

- \(S_{\text{TA}}(t)\)：脛前肌的刺激信號
- \(S_{0,\text{TA}}\)：預先設定的刺激水平
- \(G_{\text{TA}}\)：正向長度反饋的增益
- \(\ell_{\text{CE,TA}}\)：脛前肌的肌肉纖維長度
- \(\ell_{\text{off,TA}}\)：肌肉的長度偏移
- \(F_{\text{SOL}}(t - \Delta t_{\text{SOL}})\)：比目魚肌的負向力反饋，用於抑制脛前肌的活動
- \(G_{\text{SOLTA}}\)：負向力反饋的增益

# 4. 髖屈肌的反射公式 (擺動期)

$
S_{\text{HFL}}(t) = S_{0,\text{HFL}} + G_{\text{HFL}} (\ell_{\text{CE,HFL}} - \ell_{\text{off,HFL}})(t - \Delta t_{\text{HFL}}) - G_{\text{HAM,HFL}} (\ell_{\text{CE,HAM}} - \ell_{\text{off,HAM}})(t - \Delta t_{\text{HAM}})
$
- \(S_{\text{HFL}}(t)\)：髖屈肌的刺激信號
- \(S_{0,\text{HFL}}\)：預先設定的刺激水平
- \(G_{\text{HFL}}\)：正向長度反饋的增益
- \(\ell_{\text{CE,HFL}}\)：髖屈肌的肌肉纖維長度
- \(\ell_{\text{off,HFL}}\)：髖屈肌的長度偏移
- \(G_{\text{HAM,HFL}}\)：來自股二頭肌的負向長度反饋的增益
- \(\ell_{\text{CE,HAM}}\)：股二頭肌的肌肉纖維長度

# 5. 髖伸肌和股二頭肌的反射公式 (擺動期)

- 髖伸肌的刺激信號：

$
S_{\text{GLU}}(t) = S_{0,\text{GLU}} + G_{\text{GLU}} F_{\text{GLU}}(t - \Delta t_{\text{GLU}})
$

- 股二頭肌的刺激信號：

$
S_{\text{HAM}}(t) = S_{0,\text{HAM}} + G_{\text{HAM}} F_{\text{HAM}}(t - \Delta t_{\text{HAM}})
$

# 6. 對側腿負重與擺動腿的控制公式

$
S_{\text{VAS}}(t) = S_{0,\text{VAS}} + G_{\text{VAS}} F_{\text{VAS}}(t - \Delta t_{\text{VAS}}) - k_{\phi} \Delta \phi_k(t - \Delta t_k) - k_{\text{bw}} |F_{\text{contra leg}}| \cdot \text{DSup}
$

- \(S_{\text{VAS}}(t)\)：股四頭肌的刺激信號
- \(F_{\text{VAS}}(t - \Delta t_{\text{VAS}})\)：正向力反饋
- \(\Delta \phi_k\)：膝關節角度
- \(k_{\text{bw}}\)：基於對側腿承受重量的增益
- \(F_{\text{contra leg}}\)：對側腿的力
- DSup：如果模型處於雙支撐期，該值為1
