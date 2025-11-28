
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

class RecommenderDemo:
    """
    Интерактивное демо для тестирования рекомендаций
    """
    
    def __init__(self, models_dir='models'):
        print("🔧 Загружаем модели...")
        
        # Загружаем метаданные
        with open(f'{models_dir}/recommender_meta.pkl', 'rb') as f:
            meta = pickle.load(f)
            self.all_products = meta['all_products']
            self.feature_names = meta['feature_names']
            self.product_catalog = meta['product_catalog']
        
        # Загружаем scaler
        with open(f'{models_dir}/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Загружаем модели
        self.models = {}
        for product in self.all_products:
            model_path = f'{models_dir}/model_{product}.json'
            try:
                model = xgb.XGBClassifier()
                model.load_model(model_path)
                self.models[product] = model
            except:
                pass
        
        print(f"✅ Загружено {len(self.models)} моделей для {len(self.all_products)} продуктов")
        
        # Загружаем примеры пользователей
        self.sample_users = pd.read_parquet('user_features_enhanced.pq')
        
    def create_user_persona(self, persona_type):
        """
        Создаем типичные персоны клиентов
        """
        personas = {
            'молодой_активный': {
                'market_events': 120,
                'market_clicks': 35,
                'engagement_ratio': 0.29,
                'tech_interest_ratio': 0.65,
                'sports_interest_ratio': 0.25,
                'home_interest_ratio': 0.10,
                'offers_engagement': 10,
                'offers_engagement_ratio': 0.22,
                'retail_purchase_intent': 0.18
            },
            'семья_ипотека': {
                'market_events': 65,
                'market_clicks': 15,
                'engagement_ratio': 0.23,
                'tech_interest_ratio': 0.20,
                'sports_interest_ratio': 0.10,
                'home_interest_ratio': 0.75,
                'offers_engagement': 8,
                'offers_engagement_ratio': 0.18,
                'retail_purchase_intent': 0.25
            },
            'пенсионер': {
                'market_events': 15,
                'market_clicks': 3,
                'engagement_ratio': 0.20,
                'tech_interest_ratio': 0.05,
                'sports_interest_ratio': 0.05,
                'home_interest_ratio': 0.20,
                'offers_engagement': 2,
                'offers_engagement_ratio': 0.08,
                'retail_purchase_intent': 0.10
            },
            'премиум_клиент': {
                'market_events': 180,
                'market_clicks': 50,
                'engagement_ratio': 0.28,
                'tech_interest_ratio': 0.50,
                'sports_interest_ratio': 0.30,
                'home_interest_ratio': 0.20,
                'offers_engagement': 18,
                'offers_engagement_ratio': 0.30,
                'retail_purchase_intent': 0.35
            },
            'инвестор': {
                'market_events': 95,
                'market_clicks': 20,
                'engagement_ratio': 0.21,
                'tech_interest_ratio': 0.70,
                'sports_interest_ratio': 0.10,
                'home_interest_ratio': 0.15,
                'offers_engagement': 12,
                'offers_engagement_ratio': 0.25,
                'retail_purchase_intent': 0.15
            },
            'спортсмен': {
                'market_events': 85,
                'market_clicks': 25,
                'engagement_ratio': 0.29,
                'tech_interest_ratio': 0.30,
                'sports_interest_ratio': 0.65,
                'home_interest_ratio': 0.05,
                'offers_engagement': 9,
                'offers_engagement_ratio': 0.19,
                'retail_purchase_intent': 0.22
            }
        }
        
        # Заполняем все остальные фичи нулями
        persona = {feat: 0 for feat in self.feature_names}
        
        # Обновляем значимыми фичами
        if persona_type in personas:
            persona.update(personas[persona_type])
        
        return persona
    
    def recommend(self, user_features, top_n=10, min_score=0.1):
        """
        Генерируем рекомендации
        """
        # Подготовка
        X = pd.DataFrame([user_features])[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Предсказания
        scores = {}
        for product, model in self.models.items():
            try:
                proba = model.predict_proba(X_scaled)[0, 1]
                priority = self.product_catalog[product]['priority']
                boosted_score = proba * (priority / 10.0)
                
                if boosted_score > min_score:
                    scores[product] = {
                        'score': boosted_score,
                        'probability': proba,
                        'category': self.product_catalog[product]['category'],
                        'priority': priority
                    }
            except:
                pass
        
        # Сортировка
        sorted_recs = sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True)[:top_n]
        
        return sorted_recs
    
    def format_recommendations(self, recommendations):
        """
        Красивый вывод рекомендаций
        """
        print("\n" + "="*80)
        print("🎯 ПЕРСОНАЛИЗИРОВАННЫЕ РЕКОМЕНДАЦИИ")
        print("="*80)
        
        if not recommendations:
            print("❌ Нет подходящих рекомендаций")
            return
        
        for i, (product_id, data) in enumerate(recommendations, 1):
            category = data['category']
            score = data['score']
            proba = data['probability']
            
            # Эмодзи для категорий
            emoji_map = {
                'savings': '💰',
                'premium': '👑',
                'cards': '💳',
                'partner_cards': '🎁',
                'loans': '🏠',
                'investments': '📈',
                'insurance': '🛡️'
            }
            
            emoji = emoji_map.get(category, '📦')
            
            print(f"\n{i}. {emoji} {product_id.upper()}")
            print(f"   Категория: {category}")
            print(f"   Релевантность: {score:.3f}")
            print(f"   Вероятность: {proba:.1%}")
            print(f"   Приоритет: {'⭐' * data['priority']}")
    
    def analyze_coverage(self):
        """
        Анализируем покрытие продуктов
        """
        print("\n" + "="*80)
        print("📊 АНАЛИЗ ПОКРЫТИЯ ПРОДУКТОВ")
        print("="*80)
        
        # Тестируем на случайных пользователях
        sample_size = min(100, len(self.sample_users))
        all_recommended = set()
        
        for i in range(sample_size):
            user = self.sample_users.iloc[i].to_dict()
            recs = self.recommend(user, top_n=7, min_score=0.05)
            all_recommended.update([r[0] for r in recs])
        
        print(f"\n✅ Покрытие: {len(all_recommended)}/{len(self.all_products)} продуктов")
        print(f"   ({len(all_recommended)/len(self.all_products)*100:.1f}%)")
        
        # Распределение по категориям
        category_counts = {}
        for product in all_recommended:
            cat = self.product_catalog[product]['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        print("\n📋 Распределение по категориям:")
        for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {cat:20} : {count:3} продуктов")
        
        # Не рекомендуемые продукты
        not_recommended = set(self.all_products) - all_recommended
        if not_recommended:
            print(f"\n⚠️ Не рекомендуются ({len(not_recommended)}):")
            for product in sorted(not_recommended)[:10]:
                print(f"   - {product}")
    
    def compare_personas(self):
        """
        Сравниваем рекомендации для разных персон
        """
        print("\n" + "="*80)
        print("👥 СРАВНЕНИЕ РЕКОМЕНДАЦИЙ ДЛЯ РАЗНЫХ ПЕРСОН")
        print("="*80)
        
        personas = [
            'молодой_активный',
            'семья_ипотека', 
            'пенсионер',
            'премиум_клиент',
            'инвестор',
            'спортсмен'
        ]
        
        for persona_name in personas:
            user = self.create_user_persona(persona_name)
            recs = self.recommend(user, top_n=5, min_score=0.1)
            
            print(f"\n🎭 {persona_name.upper().replace('_', ' ')}")
            print("-" * 80)
            
            if recs:
                for i, (product, data) in enumerate(recs, 1):
                    print(f"   {i}. {product:30} | {data['category']:15} | {data['probability']:.1%}")
            else:
                print("   ❌ Нет рекомендаций")
    
    def test_specific_user(self, user_id=None):
        """
        Тестируем конкретного пользователя
        """
        if user_id is None:
            user_id = np.random.randint(0, len(self.sample_users))
        
        user = self.sample_users.iloc[user_id].to_dict()
        
        print("\n" + "="*80)
        print(f"🔍 АНАЛИЗ ПОЛЬЗОВАТЕЛЯ #{user_id}")
        print("="*80)
        
        # Профиль
        print("\n📊 Профиль:")
        key_features = [
            'market_events', 'market_clicks', 'engagement_ratio',
            'tech_interest_ratio', 'sports_interest_ratio', 
            'home_interest_ratio', 'offers_engagement'
        ]
        
        for feat in key_features:
            if feat in user:
                print(f"   {feat:25} : {user[feat]:.2f}")
        
        # Рекомендации
        recs = self.recommend(user, top_n=10)
        self.format_recommendations(recs)


# ===================
# ИНТЕРАКТИВНОЕ МЕНЮ
# ===================
def main():
    print("🚀 ЗАПУСК ДЕМО РЕКОМЕНДАТЕЛЬНОЙ СИСТЕМЫ")
    print("="*80)
    
    demo = RecommenderDemo()
    
    while True:
        print("\n" + "="*80)
        print("📋 МЕНЮ:")
        print("="*80)
        print("1. Тестировать случайного пользователя")
        print("2. Сравнить персоны")
        print("3. Анализ покрытия продуктов")
        print("4. Создать свою персону")
        print("0. Выход")
        
        choice = input("\nВыберите опцию: ").strip()
        
        if choice == '1':
            demo.test_specific_user()
            
        elif choice == '2':
            demo.compare_personas()
            
        elif choice == '3':
            demo.analyze_coverage()
            
        elif choice == '4':
            print("\n🎭 СОЗДАНИЕ ПЕРСОНЫ")
            print("-" * 80)
            
            persona = {feat: 0 for feat in demo.feature_names}
            
            print("\nВведите характеристики (Enter для пропуска):")
            
            try:
                market_events = input("  Активность (0-200): ")
                if market_events:
                    persona['market_events'] = float(market_events)
                
                tech = input("  Интерес к технологиям (0-1): ")
                if tech:
                    persona['tech_interest_ratio'] = float(tech)
                
                sports = input("  Интерес к спорту (0-1): ")
                if sports:
                    persona['sports_interest_ratio'] = float(sports)
                
                home = input("  Интерес к недвижимости (0-1): ")
                if home:
                    persona['home_interest_ratio'] = float(home)
                
                engagement = input("  Отклик на предложения (0-20): ")
                if engagement:
                    persona['offers_engagement'] = float(engagement)
                
                # Генерируем рекомендации
                recs = demo.recommend(persona, top_n=10)
                demo.format_recommendations(recs)
                
            except ValueError:
                print("❌ Некорректный ввод")
            
        elif choice == '0':
            print("\n👋 До свидания!")
            break
        
        else:
            print("❌ Неверный выбор")


if __name__ == "__main__":
    main()