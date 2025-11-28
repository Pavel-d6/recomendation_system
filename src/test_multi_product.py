import pandas as pd
import sys
import os

# Добавляем путь к src
sys.path.append('.')

from src.multi_product_recommender import MultiProductRecommender

def test_multi_product_recommender():
    """Тестирует мульти-продуктовую рекомендательную систему"""
    print("🧪 ТЕСТИРОВАНИЕ МУЛЬТИ-ПРОДУКТОВОЙ СИСТЕМЫ")
    print("=" * 50)
    
    # Загружаем данные
    try:
        if os.path.exists('user_features_enhanced.pq'):
            features_df = pd.read_parquet('user_features_enhanced.pq')
        elif os.path.exists('train_features.pq'):
            features_df = pd.read_parquet('train_features.pq')
        else:
            print("❌ Файлы с фичами не найдены!")
            return
        
        print(f"✅ Загружено {len(features_df)} пользователей")
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return
    
    # Обучаем систему
    print("\n🤖 ОБУЧЕНИЕ СИСТЕМЫ...")
    recommender = MultiProductRecommender()
    recommender.train(features_df)
    
    # Тестируем рекомендации
    print("\n🎯 ТЕСТИРУЕМ РЕКОМЕНДАЦИИ...")
    
    # Берем несколько разных пользователей для теста
    test_users = [
        features_df.iloc[0],  # Первый пользователь
        features_df.iloc[len(features_df)//2],  # Пользователь из середины
        features_df.iloc[-1]  # Последний пользователь
    ]
    
    for i, user_row in enumerate(test_users):
        user = user_row.to_dict()
        print(f"\n👤 Тестовый пользователь {i+1}:")
        print(f"   Активность: {user.get('market_events', 0):.0f} событий")
        print(f"   Вовлеченность: {user.get('engagement_ratio', 0):.2f}")
        
        # Общие рекомендации
        recs = recommender.recommend(user, top_n=5)
        
        if recs:
            print("   📋 Топ-5 рекомендаций:")
            for j, rec in enumerate(recs, 1):
                print(f"      {j}. {rec['product_id']} ({rec['probability']}) - {rec['category']}")
        else:
            print("   ❌ Нет рекомендаций")
    
    # Тестируем фильтрацию по категориям
    print("\n🔍 ТЕСТИРУЕМ ФИЛЬТРАЦИЮ ПО КАТЕГОРИЯМ:")
    test_user = features_df.iloc[0].to_dict()
    
    categories = ['savings', 'cards', 'loans', 'investments', 'insurance']
    for category in categories:
        recommendations = recommender.recommend(test_user, top_n=3, category_filter=category)
        if recommendations:
            products = [r['product_id'] for r in recommendations]
            print(f"   {category}: {products}")
        else:
            print(f"   {category}: нет рекомендаций")

if __name__ == "__main__":
    test_multi_product_recommender()