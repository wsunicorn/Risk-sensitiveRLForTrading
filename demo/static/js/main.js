// ============================================================================
// RL Trading Model Testing Platform - Main JavaScript
// ============================================================================

// Check system status on load
document.addEventListener('DOMContentLoaded', function() {
    checkStatus();
    setDefaultDates();
});

// ============================================================================
// Tab Navigation
// ============================================================================

function switchTab(tabName) {
    // Hide all tab panes
    document.querySelectorAll('.tab-pane').forEach(pane => {
        pane.classList.remove('active');
    });
    
    // Remove active class from all buttons
    document.querySelectorAll('.tab-button').forEach(button => {
        button.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(`${tabName}-tab`).classList.add('active');
    
    // Add active class to clicked button
    event.target.classList.add('active');
}

// ============================================================================
// System Status
// ============================================================================

async function checkStatus() {
    try {
        const response = await fetch('/api/status/');
        const data = await response.json();
        
        const statusIndicator = document.getElementById('statusIndicator');
        const statusText = document.getElementById('statusText');
        
        if (data.packages_available) {
            statusIndicator.className = 'status-indicator online';
            statusText.textContent = data.models_loaded ? 'Đã tải mô hình ✓' : 'Hệ thống sẵn sàng';
        } else {
            statusIndicator.className = 'status-indicator offline';
            statusText.textContent = 'Thiếu gói phụ thuộc';
        }
    } catch (error) {
        console.error('Lỗi khi kiểm tra trạng thái:', error);
        const statusIndicator = document.getElementById('statusIndicator');
        const statusText = document.getElementById('statusText');
        statusIndicator.className = 'status-indicator offline';
        statusText.textContent = 'Lỗi hệ thống';
    }
}

// ============================================================================
// Real-time Prediction
// ============================================================================

// Store predictions for comparison
let predictions = {
    ppo: null,
    cvar: null
};

async function getPrediction(model) {
    const symbol = document.getElementById('rt-symbol').value.toUpperCase();
    
    if (!symbol) {
        showError('rt-error', 'Vui lòng nhập mã giao dịch');
        return;
    }
    
    // Show loading
    document.getElementById('rt-loading-text').textContent = `Đang lấy tín hiệu ${model.toUpperCase()} cho ${symbol}...`;
    showLoading('rt-loading', 'rt-results', 'rt-error');
    
    try {
        const response = await fetch('/api/realtime_predict/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                symbol: symbol,
                model: model
            })
        });
        
        const data = await response.json();
        
        if (!response.ok || !data.success) {
            throw new Error(data.error || 'Không thể lấy tín hiệu');
        }
        
        // Store prediction
        predictions[model] = data;
        
        // Display results
        displayPrediction(model, data);
        
        // Show results section
        document.getElementById('rt-results').style.display = 'block';
        hideLoading('rt-loading', null);
        
        // Check if we should show comparison
        if (predictions.ppo && predictions.cvar) {
            showComparison();
        }
        
    } catch (error) {
        console.error('Error:', error);
        showError('rt-error', error.message);
        hideLoading('rt-loading', null);
    }
}

function displayPrediction(model, data) {
    const prefix = model;
    
    // Show the section
    document.getElementById(`${prefix}-section`).style.display = 'block';
    
    // Current price
    document.getElementById(`${prefix}-price`).textContent = `$${data.current_price.toFixed(2)}`;
    document.getElementById(`${prefix}-symbol`).textContent = data.symbol;
    document.getElementById(`${prefix}-timestamp`).textContent = `Cập nhật: ${new Date(data.timestamp).toLocaleString()}`;
    
    // Get the specific model data
    const modelData = data[model];
    
    // Signal
    const signal = modelData.signal;
    const signalBadge = document.getElementById(`${prefix}-signal`);
    const signalText = {BUY: 'MUA', SELL: 'BÁN', HOLD: 'GIỮ'}[signal] || signal;
    signalBadge.textContent = signalText;
    signalBadge.className = `signal-badge ${signal.toLowerCase()}`;
    
    // Action
    document.getElementById(`${prefix}-action`).textContent = `Hành động: ${modelData.action.toFixed(3)}`;
    
    // Strength
    const strength = modelData.strength;
    document.getElementById(`${prefix}-strength`).style.width = `${strength * 100}%`;
    document.getElementById(`${prefix}-strength-text`).textContent = `${(strength * 100).toFixed(1)}%`;
    
    // Analysis
    const analysis = generateAnalysis(model, signal, modelData.action, strength, data.current_price);
    document.getElementById(`${prefix}-analysis`).innerHTML = analysis;
}

function generateAnalysis(model, signal, action, strength, price) {
    let analysis = '';
    
    if (model === 'ppo') {
        analysis += '<p><strong>Cách tiếp cận trung lập rủi ro:</strong> ';
        if (signal === 'BUY') {
            analysis += `PPO nhận diện cơ hội mua với độ mạnh tín hiệu ${(strength * 100).toFixed(1)}%. `;
            analysis += 'Mô hình này tập trung tối đa hóa lợi nhuận mà không có ràng buộc rủi ro rõ ràng.';
        } else if (signal === 'SELL') {
            analysis += `PPO phát hiện tín hiệu bán với độ mạnh tín hiệu ${(strength * 100).toFixed(1)}%. `;
            analysis += 'Mô hình gợi ý giảm mức nắm giữ để tối ưu lợi nhuận kỳ vọng.';
        } else {
            analysis += 'PPO gợi ý giữ nguyên vị thế hiện tại. ';
            analysis += 'Điều kiện thị trường hiện chưa cho thấy cần hành động mạnh.';
        }
        analysis += '</p>';
    } else {
        analysis += '<p><strong>Cách tiếp cận nhạy cảm với rủi ro:</strong> ';
        if (signal === 'BUY') {
            analysis += `CVaR-PPO gợi ý mua với độ mạnh tín hiệu ${(strength * 100).toFixed(1)}%. `;
            analysis += 'Tín hiệu này đã xét rủi ro giảm giá và rủi ro đuôi.';
        } else if (signal === 'SELL') {
            analysis += `CVaR-PPO gợi ý bán với độ mạnh tín hiệu ${(strength * 100).toFixed(1)}%. `;
            analysis += 'Mô hình ưu tiên bảo vệ danh mục trước các kịch bản bất lợi và rủi ro sụt giảm.';
        } else {
            analysis += 'CVaR-PPO gợi ý giữ nguyên vị thế hiện tại. ';
            analysis += 'Phân tích có xét rủi ro chưa cho thấy cần tái cân bằng danh mục.';
        }
        analysis += '</p>';
    }
    
    return analysis;
}

function showComparison() {
    const section = document.getElementById('comparison-section');
    section.style.display = 'block';
    
    const ppoSignal = predictions.ppo.ppo.signal;
    const cvarSignal = predictions.cvar.cvar.signal;
    const signalLabels = {BUY: 'MUA', SELL: 'BÁN', HOLD: 'GIỮ'};
    
    let comparison = '<ul>';
    
    // Agreement check
    if (ppoSignal === cvarSignal) {
        comparison += `<li><strong>Đồng thuận:</strong> Cả hai mô hình đều gợi ý <strong>${signalLabels[ppoSignal] || ppoSignal}</strong>. Điều này cho thấy tín hiệu nhất quán giữa các cách tiếp cận rủi ro.</li>`;
    } else {
        comparison += `<li><strong>Khác biệt:</strong> PPO gợi ý <strong>${signalLabels[ppoSignal] || ppoSignal}</strong>, trong khi CVaR-PPO gợi ý <strong>${signalLabels[cvarSignal] || cvarSignal}</strong>.</li>`;
        
        if (ppoSignal === 'BUY' && (cvarSignal === 'HOLD' || cvarSignal === 'SELL')) {
            comparison += '<li><strong>Diễn giải:</strong> PPO thấy cơ hội lợi nhuận nhưng CVaR-PPO thận trọng hơn do xét đến rủi ro.</li>';
        } else if (cvarSignal === 'BUY' && (ppoSignal === 'HOLD' || ppoSignal === 'SELL')) {
            comparison += '<li><strong>Diễn giải:</strong> CVaR-PPO nhận diện cơ hội đã điều chỉnh theo rủi ro mà PPO không ưu tiên.</li>';
        }
    }
    
    // Strength comparison
    const ppoStrength = predictions.ppo.ppo.strength * 100;
    const cvarStrength = predictions.cvar.cvar.strength * 100;
    
    comparison += `<li><strong>Độ mạnh tín hiệu:</strong> PPO: ${ppoStrength.toFixed(1)}% | CVaR-PPO: ${cvarStrength.toFixed(1)}%</li>`;
    
    if (Math.abs(ppoStrength - cvarStrength) > 20) {
        comparison += '<li><strong>Ghi chú:</strong> Chênh lệch lớn về độ mạnh tín hiệu cho thấy hai mô hình diễn giải thị trường khác nhau.</li>';
    }
    
    comparison += '</ul>';
    
    document.getElementById('comparison-content').innerHTML = comparison;
}

// ============================================================================
// Backtest
// ============================================================================

async function runBacktest() {
    const symbol = document.getElementById('bt-symbol').value.toUpperCase();
    const startDate = document.getElementById('bt-start').value;
    const endDate = document.getElementById('bt-end').value;
    const initialBalance = parseInt(document.getElementById('bt-balance').value);
    
    if (!symbol || !startDate || !endDate) {
        showError('bt-error', 'Vui lòng điền đầy đủ thông tin');
        return;
    }
    
    // Validate dates
    if (new Date(startDate) >= new Date(endDate)) {
        showError('bt-error', 'Ngày bắt đầu phải trước ngày kết thúc');
        return;
    }
    
    // Show loading
    showLoading('bt-loading', 'bt-results', 'bt-error');
    
    try {
        const response = await fetch('/api/backtest/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                symbol,
                start_date: startDate,
                end_date: endDate,
                initial_balance: initialBalance
            })
        });
        
        const data = await response.json();
        
        if (!response.ok || !data.success) {
            throw new Error(data.error || 'Không thể chạy kiểm tra lại');
        }
        
        displayBacktestResults(data.results, data.data_info);
        hideLoading('bt-loading', 'bt-results');
        
    } catch (error) {
        console.error('Error:', error);
        showError('bt-error', error.message);
        hideLoading('bt-loading', null);
    }
}

function displayBacktestResults(results, dataInfo) {
    // Display period info
    document.getElementById('bt-period').textContent = `${dataInfo.start_date} đến ${dataInfo.end_date}`;
    document.getElementById('bt-days').textContent = dataInfo.num_days;
    document.getElementById('bt-symbol-display').textContent = dataInfo.symbol;
    
    // Create metrics table
    const tbody = document.getElementById('bt-metrics-body');
    tbody.innerHTML = '';
    
    const metrics = [
        { key: 'total_return', label: 'Tổng lợi nhuận', format: v => `${(v * 100).toFixed(2)}%` },
        { key: 'final_value', label: 'Giá trị cuối kỳ', format: v => `$${v.toFixed(2)}` },
        { key: 'sharpe_ratio', label: 'Tỷ lệ Sharpe', format: v => v.toFixed(4) },
        { key: 'max_drawdown', label: 'Mức sụt giảm tối đa', format: v => `${(v * 100).toFixed(2)}%` },
        { key: 'win_rate', label: 'Tỷ lệ thắng', format: v => `${(v * 100).toFixed(2)}%` },
        { key: 'volatility', label: 'Độ biến động', format: v => `${(v * 100).toFixed(2)}%` }
    ];
    
    metrics.forEach(metric => {
        const row = tbody.insertRow();
        row.innerHTML = `
            <td>${metric.label}</td>
            <td>${metric.format(results.ppo.metrics[metric.key])}</td>
            <td>${metric.format(results.cvar.metrics[metric.key])}</td>
            <td>${metric.format(results.buy_hold.metrics[metric.key])}</td>
        `;
    });
    
    // Create portfolio value chart
    createPortfolioChart(results);
    
    // Create comparison chart
    createComparisonChart(results);
}

function createPortfolioChart(results) {
    const ppoValues = results.ppo.history.map(h => h.portfolio_value);
    const cvarValues = results.cvar.history.map(h => h.portfolio_value);
    const bhValues = results.buy_hold.history.map(h => h.portfolio_value);
    const dates = results.ppo.history.map(h => h.date);
    
    const trace1 = {
        x: dates,
        y: ppoValues,
        type: 'scatter',
        mode: 'lines',
        name: 'PPO',
        line: { color: '#2563eb', width: 2 }
    };
    
    const trace2 = {
        x: dates,
        y: cvarValues,
        type: 'scatter',
        mode: 'lines',
        name: 'CVaR-PPO',
        line: { color: '#dc2626', width: 2 }
    };
    
    const trace3 = {
        x: dates,
        y: bhValues,
        type: 'scatter',
        mode: 'lines',
        name: 'Mua và Giữ',
        line: { color: '#10b981', width: 2 }
    };
    
    const layout = {
        title: 'Giá trị danh mục theo thời gian',
        xaxis: { title: 'Ngày' },
        yaxis: { title: 'Giá trị danh mục ($)' },
        hovermode: 'x unified',
        template: 'plotly_white'
    };
    
    Plotly.newPlot('bt-chart', [trace1, trace2, trace3], layout, {responsive: true});
}

function createComparisonChart(results) {
    const ctx = document.getElementById('bt-comparison-chart').getContext('2d');
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Tổng lợi nhuận', 'Tỷ lệ Sharpe', 'Tỷ lệ thắng'],
            datasets: [
                {
                    label: 'PPO',
                    data: [
                        results.ppo.metrics.total_return * 100,
                        results.ppo.metrics.sharpe_ratio,
                        results.ppo.metrics.win_rate * 100
                    ],
                    backgroundColor: 'rgba(37, 99, 235, 0.7)'
                },
                {
                    label: 'CVaR-PPO',
                    data: [
                        results.cvar.metrics.total_return * 100,
                        results.cvar.metrics.sharpe_ratio,
                        results.cvar.metrics.win_rate * 100
                    ],
                    backgroundColor: 'rgba(220, 38, 38, 0.7)'
                },
                {
                    label: 'Mua và Giữ',
                    data: [
                        results.buy_hold.metrics.total_return * 100,
                        results.buy_hold.metrics.sharpe_ratio,
                        results.buy_hold.metrics.win_rate * 100
                    ],
                    backgroundColor: 'rgba(16, 185, 129, 0.7)'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                title: {
                    display: true,
                    text: 'So sánh chỉ số hiệu suất'
                },
                legend: {
                    position: 'top'
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

// ============================================================================
// Model Comparison
// ============================================================================

async function loadComparison() {
    showLoading('cmp-loading', 'cmp-results', 'cmp-error');
    
    try {
        const response = await fetch('/api/compare_models/');
        const data = await response.json();
        
        if (!response.ok || !data.success) {
            throw new Error(data.error || 'Không thể tải dữ liệu so sánh');
        }
        
        displayComparison(data.comparison);
        hideLoading('cmp-loading', 'cmp-results');
        
    } catch (error) {
        console.error('Error:', error);
        showError('cmp-error', error.message);
        hideLoading('cmp-loading', null);
    }
}

function displayComparison(comparison) {
    const results = comparison.test_results;
    
    // PPO
    document.getElementById('cmp-ppo-return').textContent = `${(results.PPO.total_return * 100).toFixed(2)}%`;
    document.getElementById('cmp-ppo-sharpe').textContent = results.PPO.sharpe_ratio.toFixed(4);
    document.getElementById('cmp-ppo-drawdown').textContent = `${(results.PPO.max_drawdown * 100).toFixed(2)}%`;
    document.getElementById('cmp-ppo-winrate').textContent = `${(results.PPO.win_rate * 100).toFixed(2)}%`;
    document.getElementById('cmp-ppo-volatility').textContent = `${(results.PPO.volatility * 100).toFixed(2)}%`;
    
    // CVaR-PPO
    document.getElementById('cmp-cvar-return').textContent = `${(results['CVaR-PPO'].total_return * 100).toFixed(2)}%`;
    document.getElementById('cmp-cvar-sharpe').textContent = results['CVaR-PPO'].sharpe_ratio.toFixed(4);
    document.getElementById('cmp-cvar-drawdown').textContent = `${(results['CVaR-PPO'].max_drawdown * 100).toFixed(2)}%`;
    document.getElementById('cmp-cvar-winrate').textContent = `${(results['CVaR-PPO'].win_rate * 100).toFixed(2)}%`;
    document.getElementById('cmp-cvar-volatility').textContent = `${(results['CVaR-PPO'].volatility * 100).toFixed(2)}%`;
    
    // Buy & Hold
    document.getElementById('cmp-bh-return').textContent = `${(results['Buy & Hold'].total_return * 100).toFixed(2)}%`;
    document.getElementById('cmp-bh-sharpe').textContent = results['Buy & Hold'].sharpe_ratio.toFixed(4);
    document.getElementById('cmp-bh-drawdown').textContent = `${(results['Buy & Hold'].max_drawdown * 100).toFixed(2)}%`;
    document.getElementById('cmp-bh-winrate').textContent = `${(results['Buy & Hold'].win_rate * 100).toFixed(2)}%`;
    document.getElementById('cmp-bh-volatility').textContent = `${(results['Buy & Hold'].volatility * 100).toFixed(2)}%`;
    
    // Create radar chart
    createRadarChart(results);
}

function createRadarChart(results) {
    const ctx = document.getElementById('cmp-radar-chart').getContext('2d');
    
    // Normalize metrics for radar chart (0-100 scale)
    const normalizeReturn = v => Math.max(0, Math.min(100, v * 300)); // Scale returns
    const normalizeSharpe = v => Math.max(0, Math.min(100, v * 50)); // Scale Sharpe
    const normalizeDrawdown = v => Math.max(0, Math.min(100, (1 + v) * 100)); // Invert drawdown
    const normalizeWinRate = v => v * 100;
    const normalizeVolatility = v => Math.max(0, Math.min(100, (1 - v / 0.2) * 100)); // Invert volatility
    
    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Lợi nhuận', 'Tỷ lệ Sharpe', 'Kiểm soát rủi ro (sụt giảm)', 'Tỷ lệ thắng', 'Ổn định (biến động thấp)'],
            datasets: [
                {
                    label: 'PPO',
                    data: [
                        normalizeReturn(results.PPO.total_return),
                        normalizeSharpe(results.PPO.sharpe_ratio),
                        normalizeDrawdown(results.PPO.max_drawdown),
                        normalizeWinRate(results.PPO.win_rate),
                        normalizeVolatility(results.PPO.volatility)
                    ],
                    borderColor: 'rgba(37, 99, 235, 0.8)',
                    backgroundColor: 'rgba(37, 99, 235, 0.2)',
                    pointBackgroundColor: 'rgba(37, 99, 235, 1)'
                },
                {
                    label: 'CVaR-PPO',
                    data: [
                        normalizeReturn(results['CVaR-PPO'].total_return),
                        normalizeSharpe(results['CVaR-PPO'].sharpe_ratio),
                        normalizeDrawdown(results['CVaR-PPO'].max_drawdown),
                        normalizeWinRate(results['CVaR-PPO'].win_rate),
                        normalizeVolatility(results['CVaR-PPO'].volatility)
                    ],
                    borderColor: 'rgba(220, 38, 38, 0.8)',
                    backgroundColor: 'rgba(220, 38, 38, 0.2)',
                    pointBackgroundColor: 'rgba(220, 38, 38, 1)'
                },
                {
                    label: 'Mua và Giữ',
                    data: [
                        normalizeReturn(results['Buy & Hold'].total_return),
                        normalizeSharpe(results['Buy & Hold'].sharpe_ratio),
                        normalizeDrawdown(results['Buy & Hold'].max_drawdown),
                        normalizeWinRate(results['Buy & Hold'].win_rate),
                        normalizeVolatility(results['Buy & Hold'].volatility)
                    ],
                    borderColor: 'rgba(16, 185, 129, 0.8)',
                    backgroundColor: 'rgba(16, 185, 129, 0.2)',
                    pointBackgroundColor: 'rgba(16, 185, 129, 1)'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        stepSize: 20
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Biểu đồ radar hiệu suất mô hình (chỉ số đã chuẩn hóa)'
                },
                legend: {
                    position: 'top'
                }
            }
        }
    });
}

// ============================================================================
// Utility Functions
// ============================================================================

function showLoading(loadingId, resultsId, errorId) {
    document.getElementById(loadingId).style.display = 'block';
    if (resultsId) document.getElementById(resultsId).style.display = 'none';
    if (errorId) document.getElementById(errorId).style.display = 'none';
}

function hideLoading(loadingId, resultsId) {
    document.getElementById(loadingId).style.display = 'none';
    if (resultsId) document.getElementById(resultsId).style.display = 'block';
}

function showError(errorId, message) {
    const errorEl = document.getElementById(errorId);
    errorEl.textContent = message;
    errorEl.style.display = 'block';
}

function setDefaultDates() {
    const endDate = new Date();
    const startDate = new Date();
    startDate.setFullYear(startDate.getFullYear() - 1);
    
    document.getElementById('bt-end').valueAsDate = endDate;
    document.getElementById('bt-start').valueAsDate = startDate;
}

// Format number as currency
function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(value);
}

// Format percentage
function formatPercent(value) {
    return `${(value * 100).toFixed(2)}%`;
}
