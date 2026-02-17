async function postJson(url, data) {
  const token = document.querySelector('meta[name="csrf-token"]').content;
  const res = await fetch(url, { 
    method: 'POST', 
    headers: { 
      'Content-Type': 'application/json',
      'X-CSRFToken': token
    }, 
    body: JSON.stringify(data) 
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function formToJson(form) {
  const data = {};
  new FormData(form).forEach((v, k) => { data[k] = Number(v); });
  return data;
}

const shapCtx = document.getElementById('shapChart');
let shapChart = null;
let shapChartData = { labels: [], values: [] };
let currentShapChartType = 'bar';

function renderShapChart() {
  if (!shapChartData.labels.length) return;
  
  const { labels, values } = shapChartData;
  const chartType = currentShapChartType;
  
  if (shapChart) shapChart.destroy();
  
  const colors = [
    '#0d6efd', '#198754', '#dc3545', '#ffc107', '#17a2b8',
    '#6f42c1', '#e83e8c', '#fd7e14', '#20c997', '#6c757d'
  ];
  
  const chartConfig = {
    bar: {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: 'SHAP Value',
          data: values,
          backgroundColor: '#0d6efd'
        }]
      },
      options: { responsive: true, scales: { y: { beginAtZero: true } } }
    },
    horizontalBar: {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: 'SHAP Value',
          data: values,
          backgroundColor: '#0d6efd'
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        scales: { x: { beginAtZero: true } }
      }
    },
    pie: {
      type: 'pie',
      data: {
        labels,
        datasets: [{
          label: 'SHAP Value',
          data: values.map(v => Math.abs(v)),
          backgroundColor: colors.slice(0, labels.length)
        }]
      },
      options: { responsive: true }
    }
  };
  
  shapChart = new Chart(shapCtx, chartConfig[chartType]);
}

function updateShapChartType(type) {
  currentShapChartType = type;
  renderShapChart();
}

const forecastCtx = document.getElementById('forecastChart');
let forecastChart = null;
const previousTrendCtx = document.getElementById('previousTrendChart');
let previousTrendChart = null;

async function updatePredictionsList() {
  try {
    const res = await fetch('/api/predictions/recent');
    const data = await res.json();
    const predictionsList = document.querySelector('#predictions-list');
    if (!predictionsList) return;
    
    if (data.predictions && data.predictions.length > 0) {
      predictionsList.innerHTML = data.predictions.map(p => {
        const date = new Date(p.created_at);
        const dateStr = date.toLocaleString('en-US', { 
          year: 'numeric', 
          month: '2-digit', 
          day: '2-digit', 
          hour: '2-digit', 
          minute: '2-digit' 
        });
        const riskClass = p.prediction == 1 ? 'danger' : 'success';
        const riskText = p.prediction == 1 ? 'High Risk' : 'Low Risk';
        const probPercent = (p.probability * 100).toFixed(1);
        
        return `
          <li class="list-group-item d-flex justify-content-between align-items-center">
            <div>
              <i class="bi bi-calendar-check me-2 text-primary"></i>
              <strong>${dateStr}</strong>
            </div>
            <div>
              <span class="badge text-bg-${riskClass} me-2">${riskText}</span>
              <small class="text-muted">${probPercent}%</small>
            </div>
          </li>
        `;
      }).join('');
    } else {
      predictionsList.innerHTML = `
        <li class="list-group-item text-center text-muted py-4">
          <i class="bi bi-inbox me-2"></i>No predictions yet.
        </li>
      `;
    }
  } catch (error) {
    console.error('Error updating predictions list:', error);
  }
}

async function handlePredict(e) {
  e.preventDefault();
  const payload = formToJson(document.getElementById('predict-form'));
  const pred = await postJson('/api/predict', payload);
  document.getElementById('risk-label').textContent = `Risk: ${pred.prediction} (p=${pred.probability.toFixed(2)})`;
  
  // Update predictions list immediately
  await updatePredictionsList();
  
  const exp = await postJson('/api/explain', payload);
  const labels = exp.features;
  const vals = labels.map(f => exp.shap_values[f]);
  
  // Store SHAP data and render with current chart type
  shapChartData = { labels, values: vals };
  renderShapChart();
}

async function generateRecommendations(avgGlucose, minGlucose, maxGlucose, currentGlucose) {
  const dietaryGuidelines = document.getElementById('dietaryGuidelines');
  const suggestedFoods = document.getElementById('suggestedFoods');
  const recommendationsSection = document.getElementById('recommendationsSection');
  
  let guidelines = [];
  let foods = [];
  let highRisk = avgGlucose > 180 || maxGlucose > 200;
  let normalRisk = avgGlucose >= 100 && avgGlucose <= 180;
  let lowRisk = avgGlucose < 100;
  
  // Get prediction count to rotate variants
  try {
    const token = document.querySelector('meta[name="csrf-token"]').content;
    const prevRes = await fetch('/api/predictions', {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': token
      }
    });
    if (!prevRes.ok) throw new Error('Failed to fetch predictions');
    const prevData = await prevRes.json();
    const predictionCount = Array.isArray(prevData) ? prevData.length : (prevData.predictions ? prevData.predictions.length : 0);
    const variant = predictionCount % 3; // Cycles through 0, 1, 2
    
    // HIGH RISK RECOMMENDATIONS - 3 VARIANTS
    if (highRisk) {
      if (variant === 0) {
        guidelines.push('⚠️ <strong>High Glucose Risk</strong> - Reduce sugar and refined carbohydrates');
        guidelines.push('🚫 Avoid sugary drinks, desserts, and processed foods');
        guidelines.push('⏰ Consider eating at regular intervals to stabilize glucose');
        guidelines.push('💊 Consult your healthcare provider if levels remain high');
        
        foods.push('🥦 <strong>Best Choices:</strong>');
        foods.push('• Leafy greens (spinach, kale, broccoli)');
        foods.push('• Non-starchy vegetables (zucchini, bell peppers)');
        foods.push('• Lean proteins (chicken, fish, tofu)');
        foods.push('• Whole grains in moderation');
        foods.push('<br>');
        foods.push('❌ <strong>Avoid:</strong>');
        foods.push('• White bread, rice, and pasta');
        foods.push('• Sugary drinks and juices');
        foods.push('• Processed snacks and desserts');
        foods.push('• High-fat dairy products');
      } else if (variant === 1) {
        guidelines.push('🛑 <strong>Alert: Elevated Glucose Levels</strong> - Immediate dietary adjustment needed');
        guidelines.push('⚡ Focus on protein and fiber-rich foods for sustained energy');
        guidelines.push('💧 Increase water intake and reduce liquid calories');
        guidelines.push('🏃 Light physical activity after meals helps lower glucose');
        
        foods.push('🥒 <strong>Smart Swaps:</strong>');
        foods.push('• Replace rice with cauliflower rice');
        foods.push('• Choose quinoa over regular pasta');
        foods.push('• Snack on almonds instead of crackers');
        foods.push('• Pick herbal tea over sugary beverages');
        foods.push('<br>');
        foods.push('⛔ <strong>Eliminate:</strong>');
        foods.push('• All sugary snacks and beverages');
        foods.push('• Fast food and fried items');
        foods.push('• Store-bought pastries and cakes');
        foods.push('• High-calorie condiments');
      } else {
        guidelines.push('🚨 <strong>Critical: Manage Glucose Spikes</strong> - Structured meal plan required');
        guidelines.push('🔄 Space meals 3-4 hours apart for better control');
        guidelines.push('📋 Track your meals and monitor glucose response');
        guidelines.push('👨‍⚕️ Consider consulting a nutritionist or dietitian');
        
        foods.push('🌱 <strong>Recommended Foods:</strong>');
        foods.push('• Grilled chicken breast with steamed vegetables');
        foods.push('• Baked salmon with asparagus');
        foods.push('• Lentil soup with low-sodium broth');
        foods.push('• Greek yogurt (unsweetened)');
        foods.push('<br>');
        foods.push('🚫 <strong>Foods to Eliminate:</strong>');
        foods.push('• Corn, potatoes, carrots (high glycemic)');
        foods.push('• Fruit juices and smoothies');
        foods.push('• Alcohol and regular sodas');
        foods.push('• Trans fats and processed oils');
      }
    }
    // NORMAL RISK RECOMMENDATIONS - 3 VARIANTS
    else if (normalRisk) {
      if (variant === 0) {
        guidelines.push('✓ <strong>Moderate Glucose Levels</strong> - Maintain balanced diet');
        guidelines.push('🥗 Continue with balanced meals and portion control');
        guidelines.push('🚶 Regular physical activity helps manage glucose');
        guidelines.push('📊 Monitor trends over time');
        
        foods.push('🥗 <strong>Balanced Options:</strong>');
        foods.push('• Whole wheat bread and grains');
        foods.push('• Lean proteins with vegetables');
        foods.push('• Fresh fruits (berries preferred)');
        foods.push('• Nuts and seeds in moderation');
        foods.push('<br>');
        foods.push('💡 <strong>Portion Tips:</strong>');
        foods.push('• 1/4 plate proteins, 1/2 plate vegetables');
        foods.push('• 1/4 plate whole grains or starchy veggies');
        foods.push('• Stay hydrated with water');
      } else if (variant === 1) {
        guidelines.push('😊 <strong>Good Job! Glucose Under Control</strong> - Keep your routine');
        guidelines.push('🎯 Maintain consistent meal times for stability');
        guidelines.push('🏋️ Exercise 30 mins daily for optimal glucose management');
        guidelines.push('✅ Your current lifestyle is working well');
        
        foods.push('🍎 <strong>Enjoy These Foods:</strong>');
        foods.push('• Brown rice and oats');
        foods.push('• Olive oil-based dressings');
        foods.push('• Chicken, turkey, and lean beef');
        foods.push('• All vegetables and most fruits');
        foods.push('<br>');
        foods.push('🎉 <strong>Healthy Indulgences:</strong>');
        foods.push('• Dark chocolate (70% cocoa or higher)');
        foods.push('• Whole grain crackers with cheese');
        foods.push('• Homemade smoothies with fiber');
        foods.push('• Nuts and seeds as snacks');
      } else {
        guidelines.push('⭐ <strong>On Track! Glucose Levels Stable</strong> - Continue current habits');
        guidelines.push('🎪 Balance is key - mix proteins, fats, and carbs');
        guidelines.push('💪 Strength training helps improve glucose sensitivity');
        guidelines.push('📱 Keep monitoring to maintain consistency');
        
        foods.push('🍱 <strong>Sample Daily Menu:</strong>');
        foods.push('• Breakfast: Oatmeal with berries and nuts');
        foods.push('• Lunch: Grilled chicken with brown rice');
        foods.push('• Snack: Apple with almond butter');
        foods.push('• Dinner: Baked fish with quinoa and veggies');
        foods.push('<br>');
        foods.push('✨ <strong>Pro Tips:</strong>');
        foods.push('• Eat slowly and mindfully');
        foods.push('• Combine proteins with each meal');
        foods.push('• Drink water before eating');
        foods.push('• Avoid eating late at night');
      }
    }
    // LOW RISK RECOMMENDATIONS - 3 VARIANTS
    else if (lowRisk) {
      if (variant === 0) {
        guidelines.push('✓ <strong>Healthy Glucose Levels</strong> - Keep up the good work!');
        guidelines.push('🎯 Continue monitoring and maintain healthy habits');
        guidelines.push('😊 Your lifestyle choices are working well');
        guidelines.push('🏆 You are doing an excellent job!');
        
        foods.push('🌟 <strong>Continue With:</strong>');
        foods.push('• Healthy proteins and whole grains');
        foods.push('• Plenty of fresh vegetables and fruits');
        foods.push('• Nuts, seeds, and healthy fats');
        foods.push('• Regular balanced meals');
      } else if (variant === 1) {
        guidelines.push('🎉 <strong>Excellent! Glucose at Optimal Levels</strong> - Amazing progress!');
        guidelines.push('🌟 Your dedication to health is paying off');
        guidelines.push('💚 Keep following your nutritious eating pattern');
        guidelines.push('🙌 You\'re a great example of healthy living');
        
        foods.push('👑 <strong>Premium Food Choices:</strong>');
        foods.push('• Wild salmon with herbs and lemon');
        foods.push('• Organic vegetables and locally sourced produce');
        foods.push('• Grass-fed lean meats');
        foods.push('• Superfoods: quinoa, chia, acai, goji berries');
        foods.push('<br>');
        foods.push('🌈 <strong>Variety Ideas:</strong>');
        foods.push('• Try new healthy recipes weekly');
        foods.push('• Explore different cuisines (Mediterranean, Asian)');
        foods.push('• Experiment with herbs and spices');
        foods.push('• Share recipes with friends and family');
      } else {
        guidelines.push('🏅 <strong>Peak Performance! Your Glucose is Perfect</strong> - Fantastic!');
        guidelines.push('⚡ Your metabolism is working optimally');
        guidelines.push('🎊 Continue this winning momentum');
        guidelines.push('💪 You\'ve mastered diabetes management!');
        
        foods.push('🥇 <strong>Superstar Meals:</strong>');
        foods.push('• Mediterranean bowls with olive oil');
        foods.push('• Smoothie bowls with chia and granola');
        foods.push('• Vegetable stir-fries with tofu or shrimp');
        foods.push('• Homemade nutritious snack bars');
        foods.push('<br>');
        foods.push('🎯 <strong>Advanced Wellness:</strong>');
        foods.push('• Meal prep for consistent nutrition');
        foods.push('• Try intermittent fasting if interested');
        foods.push('• Explore plant-based meals');
        foods.push('• Share your success story to inspire others');
      }
    }
  } catch (error) {
    console.error('Error fetching prediction count:', error);
  }
  
  // Populate recommendations
  dietaryGuidelines.innerHTML = guidelines.map(g => `<div class="mb-2">${g}</div>`).join('');
  suggestedFoods.innerHTML = foods.map(f => `<div class="mb-1">${f}</div>`).join('');
  recommendationsSection.style.display = 'block';
}

async function handleAutoForecast() {
  const forecastInfo = document.getElementById('forecastInfo');
  const forecastMessage = document.getElementById('forecastMessage');
  const trendAnalysis = document.getElementById('trendAnalysis');
  const trendMessage = document.getElementById('trendMessage');
  const glucose = parseFloat(document.querySelector('input[name="Glucose"]').value) || 120;
  
  forecastInfo.style.display = 'block';
  forecastMessage.textContent = `Predicting glucose levels for the next hour from current reading (${glucose} mg/dL)...`;
  
  try {
    // Fetch current forecast
    const res = await postJson('/api/forecast', { readings: [] });
    const pts = res.forecast;
    
    if (!pts || pts.length === 0) {
      forecastMessage.textContent = 'No forecast data available. Try uploading CGM data.';
      return;
    }
    
    const labels = pts.map((p, i) => {
      const min = i * 6;
      return `${Math.floor(min / 60)}:${String(min % 60).padStart(2, '0')}`;
    });
    const vals = pts.map(p => p.glucose);
    const avgGlucose = (vals.reduce((a, b) => a + b, 0) / vals.length).toFixed(1);
    const minGlucose = Math.min(...vals).toFixed(1);
    const maxGlucose = Math.max(...vals).toFixed(1);
    
    // Fetch previous predictions for trend
    const prevRes = await fetch('/api/predictions/recent');
    const prevData = await prevRes.json();
    let previousAvg = 0;
    if (prevData.predictions && prevData.predictions.length > 0) {
      const previousProbs = prevData.predictions.slice(0, 5).map(p => p.probability * 100);
      previousAvg = (previousProbs.reduce((a, b) => a + b, 0) / previousProbs.length).toFixed(1);
    }
    
    forecastMessage.innerHTML = `<strong>Next Hour Forecast:</strong> Avg: ${avgGlucose} mg/dL | Range: ${minGlucose}-${maxGlucose} mg/dL`;
    
    // Current Forecast Chart
    if (forecastChart) forecastChart.destroy();
    forecastChart = new Chart(forecastCtx, {
      type: 'line',
      data: {
        labels,
        datasets: [{
          label: 'Predicted Glucose (mg/dL)',
          data: vals,
          borderColor: '#28a745',
          backgroundColor: 'rgba(40, 167, 69, 0.1)',
          fill: true,
          tension: 0.4,
          pointBackgroundColor: '#28a745',
          pointBorderColor: '#fff',
          pointBorderWidth: 2
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: true }
        },
        scales: {
          y: {
            beginAtZero: false,
            ticks: { callback: function(value) { return value + ' mg/dL'; } }
          }
        }
      }
    });
    
    // Previous Trend Chart - Show last 5 predictions
    if (previousTrendChart) previousTrendChart.destroy();
    if (prevData.predictions && prevData.predictions.length > 0) {
      const lastPredictions = prevData.predictions.slice(0, 5).reverse();
      const predictionLabels = lastPredictions.map((p, i) => `${i + 1}h ago`);
      const predictionProbs = lastPredictions.map(p => (p.probability * 100).toFixed(1));
      const trendColor = parseFloat(predictionProbs[predictionProbs.length - 1]) > parseFloat(predictionProbs[0]) ? '#dc3545' : '#28a745';
      
      previousTrendChart = new Chart(previousTrendCtx, {
        type: 'line',
        data: {
          labels: predictionLabels,
          datasets: [{
            label: 'Risk Probability (%)',
            data: predictionProbs,
            borderColor: trendColor,
            backgroundColor: trendColor === '#dc3545' ? 'rgba(220, 53, 69, 0.1)' : 'rgba(40, 167, 69, 0.1)',
            fill: true,
            tension: 0.4,
            pointBackgroundColor: trendColor,
            pointBorderColor: '#fff',
            pointBorderWidth: 2
          }]
        },
        options: {
          responsive: true,
          plugins: {
            legend: { display: true }
          },
          scales: {
            y: {
              beginAtZero: true,
              max: 100,
              ticks: { callback: function(value) { return value + '%'; } }
            }
          }
        }
      });
      
      // Show trend analysis
      const isIncreasing = parseFloat(predictionProbs[predictionProbs.length - 1]) > parseFloat(predictionProbs[0]);
      const trendIcon = isIncreasing ? '📈' : '📉';
      const trendText = isIncreasing ? 'increasing' : 'decreasing';
      trendMessage.innerHTML = `${trendIcon} <strong>Risk Trend:</strong> Your glucose risk is ${trendText} compared to previous predictions. ${isIncreasing ? '⚠️ Pay attention to your diet!' : '✓ Good! Keep maintaining your healthy habits!'}`;
      trendAnalysis.style.display = 'block';
    }
    
    // Generate personalized recommendations
    await generateRecommendations(avgGlucose, minGlucose, maxGlucose, glucose);
  } catch (error) {
    forecastMessage.textContent = 'Error generating forecast: ' + error.message;
    console.error('Forecast error:', error);
  }
}

async function handleForecast() {
  const res = await postJson('/api/forecast', { readings: [] });
  const pts = res.forecast;
  const labels = pts.map(p => new Date(p.timestamp).toLocaleTimeString());
  const vals = pts.map(p => p.glucose);
  if (forecastChart) forecastChart.destroy();
  forecastChart = new Chart(forecastCtx, {
    type: 'line',
    data: { labels, datasets: [{ label: 'Glucose', data: vals, borderColor: '#198754' }] },
    options: { responsive: true }
  });
}

async function handleUpload() {
  const f = document.getElementById('cgmFile').files[0];
  if (!f) return;
  const form = new FormData();
  form.append('file', f);
  const res = await fetch('/api/forecast', { method: 'POST', body: form });
  const data = await res.json();
  const pts = data.forecast;
  const labels = pts.map(p => new Date(p.timestamp).toLocaleTimeString());
  const vals = pts.map(p => p.glucose);
  if (forecastChart) forecastChart.destroy();
  forecastChart = new Chart(forecastCtx, {
    type: 'line',
    data: { labels, datasets: [{ label: 'Glucose', data: vals, borderColor: '#198754' }] },
    options: { responsive: true }
  });
}

window.addEventListener('DOMContentLoaded', () => {
  const pf = document.getElementById('predict-form');
  if (pf) pf.addEventListener('submit', handlePredict);
  const afb = document.getElementById('autoForecastBtn');
  if (afb) afb.addEventListener('click', handleAutoForecast);
  
  // Add event listeners for SHAP chart type selection
  const shapTypeRadios = document.querySelectorAll('input[name="shapChartType"]');
  shapTypeRadios.forEach(radio => {
    radio.addEventListener('change', (e) => {
      updateShapChartType(e.target.value);
    });
  });
  
  // Update predictions list on page load (in case of new predictions)
  if (document.getElementById('predictions-list')) {
    updatePredictionsList();
  }
});







