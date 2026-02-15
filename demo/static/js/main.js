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
        const response = await fetch('/api/status');
        const data = await response.json();
        
        const statusIndicator = document.getElementById('statusIndicator');
        const statusText = document.getElementById('statusText');
        
        if (data.packages_available) {
            statusIndicator.className = 'status-indicator online';
            statusText.textContent = data.models_loaded ? 'Models Loaded ✓' : 'System Ready';
        } else {
            statusIndicator.className = 'status-indicator offline';
            statusText.textContent = 'Packages Not Installed';
        }
    } catch (error) {
        console.error('Error checking status:', error);
        const statusIndicator = document.getElementById('statusIndicator');
        const statusText = document.getElementById('statusText');
        statusIndicator.className = 'status-indicator offline';
        statusText.textContent = 'System Error';
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
        showError('rt-error', 'Please enter a stock symbol');
        return;
    }
    
    // Show loading
    document.getElementById('rt-loading-text').textContent = `Fetching ${model.toUpperCase()} prediction for ${symbol}...`;
    showLoading('rt-loading', 'rt-results', 'rt-error');
    
    try {
        const response = await fetch('/api/realtime_predict', {
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
            throw new Error(data.error || 'Failed to get prediction');
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
    document.getElementById(`${prefix}-timestamp`).textContent = `Updated: ${new Date(data.timestamp).toLocaleString()}`;
    
    // Get the specific model data
    const modelData = data[model];
    
    // Signal
    const signal = modelData.signal;
    const signalBadge = document.getElementById(`${prefix}-signal`);
    signalBadge.textContent = signal;
    signalBadge.className = `signal-badge ${signal.toLowerCase()}`;
    
    // Action
    document.getElementById(`${prefix}-action`).textContent = `Action: ${modelData.action.toFixed(3)}`;
    
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
        analysis += '<p><strong>Risk-Neutral Approach:</strong> ';
        if (signal === 'BUY') {
            analysis += `PPO identifies a bullish opportunity with ${(strength * 100).toFixed(1)}% confidence. `;
            analysis += 'This model focuses on maximizing returns without explicit risk constraints.';
        } else if (signal === 'SELL') {
            analysis += `PPO detects a bearish signal with ${(strength * 100).toFixed(1)}% confidence. `;
            analysis += 'The model suggests reducing exposure to maximize expected returns.';
        } else {
            analysis += 'PPO recommends holding current position. ';
            analysis += 'Market conditions do not warrant significant action at this time.';
        }
        analysis += '</p>';
    } else {
        analysis += '<p><strong>Risk-Sensitive Approach:</strong> ';
        if (signal === 'BUY') {
            analysis += `CVaR-PPO suggests buying with ${(strength * 100).toFixed(1)}% confidence. `;
            analysis += 'This signal considers downside risk and tail events, indicating acceptable risk-adjusted opportunity.';
        } else if (signal === 'SELL') {
            analysis += `CVaR-PPO recommends selling with ${(strength * 100).toFixed(1)}% confidence. `;
            analysis += 'The model prioritizes protecting against worst-case scenarios and managing drawdown risk.';
        } else {
            analysis += 'CVaR-PPO advises holding current position. ';
            analysis += 'Risk-adjusted analysis does not justify portfolio rebalancing at this moment.';
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
    
    let comparison = '<ul>';
    
    // Agreement check
    if (ppoSignal === cvarSignal) {
        comparison += `<li><strong>✅ Agreement:</strong> Both models recommend <strong>${ppoSignal}</strong>. This suggests strong conviction across different risk approaches.</li>`;
    } else {
        comparison += `<li><strong>⚠️ Disagreement:</strong> PPO suggests <strong>${ppoSignal}</strong> while CVaR-PPO suggests <strong>${cvarSignal}</strong>.</li>`;
        
        if (ppoSignal === 'BUY' && (cvarSignal === 'HOLD' || cvarSignal === 'SELL')) {
            comparison += '<li><strong>Interpretation:</strong> PPO sees profit opportunity but CVaR-PPO is more cautious due to risk considerations.</li>';
        } else if (cvarSignal === 'BUY' && (ppoSignal === 'HOLD' || ppoSignal === 'SELL')) {
            comparison += '<li><strong>Interpretation:</strong> CVaR-PPO finds risk-adjusted opportunity that PPO does not prioritize.</li>';
        }
    }
    
    // Strength comparison
    const ppoStrength = predictions.ppo.ppo.strength * 100;
    const cvarStrength = predictions.cvar.cvar.strength * 100;
    
    comparison += `<li><strong>Confidence Levels:</strong> PPO: ${ppoStrength.toFixed(1)}% | CVaR-PPO: ${cvarStrength.toFixed(1)}%</li>`;
    
    if (Math.abs(ppoStrength - cvarStrength) > 20) {
        comparison += '<li><strong>Note:</strong> Significant difference in confidence levels suggests different market interpretations.</li>';
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
        showError('bt-error', 'Please fill in all fields');
        return;
    }
    
    // Validate dates
    if (new Date(startDate) >= new Date(endDate)) {
        showError('bt-error', 'Start date must be before end date');
        return;
    }
    
    // Show loading
    showLoading('bt-loading', 'bt-results', 'bt-error');
    
    try {
        const response = await fetch('/api/backtest', {
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
            throw new Error(data.error || 'Failed to run backtest');
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
    document.getElementById('bt-period').textContent = `${dataInfo.start_date} to ${dataInfo.end_date}`;
    document.getElementById('bt-days').textContent = dataInfo.num_days;
    document.getElementById('bt-symbol-display').textContent = dataInfo.symbol;
    
    // Create metrics table
    const tbody = document.getElementById('bt-metrics-body');
    tbody.innerHTML = '';
    
    const metrics = [
        { key: 'total_return', label: 'Total Return', format: v => `${(v * 100).toFixed(2)}%` },
        { key: 'final_value', label: 'Final Value', format: v => `$${v.toFixed(2)}` },
        { key: 'sharpe_ratio', label: 'Sharpe Ratio', format: v => v.toFixed(4) },
        { key: 'max_drawdown', label: 'Max Drawdown', format: v => `${(v * 100).toFixed(2)}%` },
        { key: 'win_rate', label: 'Win Rate', format: v => `${(v * 100).toFixed(2)}%` },
        { key: 'volatility', label: 'Volatility', format: v => `${(v * 100).toFixed(2)}%` }
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
        name: 'Buy & Hold',
        line: { color: '#10b981', width: 2 }
    };
    
    const layout = {
        title: 'Portfolio Value Over Time',
        xaxis: { title: 'Date' },
        yaxis: { title: 'Portfolio Value ($)' },
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
            labels: ['Total Return', 'Sharpe Ratio', 'Win Rate'],
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
                    label: 'Buy & Hold',
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
                    text: 'Performance Metrics Comparison'
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
        const response = await fetch('/api/compare_models');
        const data = await response.json();
        
        if (!response.ok || !data.success) {
            throw new Error(data.error || 'Failed to load comparison');
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
            labels: ['Return', 'Sharpe Ratio', 'Risk Control (Drawdown)', 'Win Rate', 'Stability (Low Vol)'],
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
                    label: 'Buy & Hold',
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
                    text: 'Model Performance Radar Chart (Normalized Metrics)'
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
