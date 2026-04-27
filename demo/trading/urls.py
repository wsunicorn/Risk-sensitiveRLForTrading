from django.urls import path
from . import views

app_name = 'trading'

urlpatterns = [
    path('', views.index, name='index'),
    path('ppo/', views.ppo_page, name='ppo_page'),
    path('cvar/', views.cvar_page, name='cvar_page'),
    path('sortino/', views.sortino_page, name='sortino_page'),
    path('compare/', views.compare_page, name='compare_page'),
    path('dashboard/', views.dashboard_page, name='dashboard_page'),
    path('api/test_models/', views.TestModelsView.as_view(), name='test_models'),
    path('api/backtest/', views.BacktestView.as_view(), name='backtest'),
    path('api/realtime_predict/', views.RealtimePredictView.as_view(), name='realtime_predict'),
    path('api/compare_models/', views.compare_models, name='compare_models'),
    path('api/status/', views.status, name='status'),
]