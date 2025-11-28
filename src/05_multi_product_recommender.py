import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss, jaccard_score
import pickle
import warnings
import os 
warnings.filterwarnings('ignore')

# ========================
# ПОЛНЫЙ КАТАЛОГ ПРОДУКТОВ
# ========================
FULL_PRODUCT_CATALOG = {
    # ВКЛАДЫ (5 продуктов)
    'deposit_savings': {'category': 'savings', 'priority': 8, 'min_age': 18},
    'deposit_profitable': {'category': 'savings', 'priority': 9, 'min_age': 18},
    'deposit_pension': {'category': 'savings', 'priority': 7, 'min_age': 55},
    'deposit_special': {'category': 'savings', 'priority': 10, 'min_age': 18},
    'savings_free': {'category': 'savings', 'priority': 8, 'min_age': 18},
    
    # ПРЕМИУМ (3 продукта)
    'premium_card': {'category': 'premium', 'priority': 10, 'min_age': 25},
    'premium_package': {'category': 'premium', 'priority': 10, 'min_age': 30},
    'premium_investment': {'category': 'premium', 'priority': 9, 'min_age': 30},
    
    # КАРТЫ (12 продуктов)
    'credit_card_180': {'category': 'cards', 'priority': 9, 'min_age': 21},
    'salary_card_pro': {'category': 'cards', 'priority': 8, 'min_age': 18},
    'sports_card': {'category': 'cards', 'priority': 7, 'min_age': 18},
    'pension_card': {'category': 'cards', 'priority': 6, 'min_age': 55},
    'card_strong_people': {'category': 'cards', 'priority': 9, 'min_age': 21},
    'card_resident': {'category': 'cards', 'priority': 6, 'min_age': 18},
    'card_cashback': {'category': 'cards', 'priority': 8, 'min_age': 18},
    'card_salary_plus': {'category': 'cards', 'priority': 7, 'min_age': 18},
    'card_psb_iz': {'category': 'cards', 'priority': 8, 'min_age': 21},
    
    # ПАРТНЕРСКИЕ КАРТЫ (9 продуктов)
    'card_spartak': {'category': 'partner_cards', 'priority': 7, 'min_age': 18},
    'card_cska': {'category': 'partner_cards', 'priority': 7, 'min_age': 18},
    'card_lenta': {'category': 'partner_cards', 'priority': 8, 'min_age': 18},
    'card_vkusvill': {'category': 'partner_cards', 'priority': 7, 'min_age': 18},
    'card_sportmaster': {'category': 'partner_cards', 'priority': 7, 'min_age': 18},
    'card_mvideo': {'category': 'partner_cards', 'priority': 8, 'min_age': 18},
    'card_post_market': {'category': 'partner_cards', 'priority': 6, 'min_age': 18},
    'card_new_world': {'category': 'partner_cards', 'priority': 6, 'min_age': 18},
    
    # КРЕДИТЫ И ИПОТЕКА (15 продуктов)
    'consumer_loan': {'category': 'loans', 'priority': 9, 'min_age': 21},
    'refinancing': {'category': 'loans', 'priority': 8, 'min_age': 23},
    'mortgage_new': {'category': 'loans', 'priority': 10, 'min_age': 21},
    'mortgage_family': {'category': 'loans', 'priority': 10, 'min_age': 21},
    'mortgage_military': {'category': 'loans', 'priority': 9, 'min_age': 20},
    'mortgage_far_east': {'category': 'loans', 'priority': 8, 'min_age': 21},
    'mortgage_alternative': {'category': 'loans', 'priority': 7, 'min_age': 25},
    'mortgage_secondary': {'category': 'loans', 'priority': 9, 'min_age': 21},
    'mortgage_castle': {'category': 'loans', 'priority': 8, 'min_age': 25},
    'mortgage_easy': {'category': 'loans', 'priority': 7, 'min_age': 23},
    
    # ИНВЕСТИЦИИ (12 продуктов)
    'investment_stocks': {'category': 'investments', 'priority': 8, 'min_age': 25},
    'investment_bonds': {'category': 'investments', 'priority': 7, 'min_age': 25},
    'investment_mixed': {'category': 'investments', 'priority': 7, 'min_age': 25},
    'investment_defense': {'category': 'investments', 'priority': 8, 'min_age': 25},
    'investment_dividend': {'category': 'investments', 'priority': 7, 'min_age': 25},
    'investment_perspective': {'category': 'investments', 'priority': 8, 'min_age': 25},
    'investment_opportunities': {'category': 'investments', 'priority': 7, 'min_age': 25},
    'investment_world': {'category': 'investments', 'priority': 6, 'min_age': 30},
    'investment_cushion': {'category': 'investments', 'priority': 6, 'min_age': 23},
    'investment_flow': {'category': 'investments', 'priority': 7, 'min_age': 25},
    'investment_resources': {'category': 'investments', 'priority': 7, 'min_age': 25},
    'investment_east': {'category': 'investments', 'priority': 6, 'min_age': 25},
    
    # СТРАХОВАНИЕ (14 продуктов)
    'insurance_osago': {'category': 'insurance', 'priority': 9, 'min_age': 18},
    'insurance_job_loss': {'category': 'insurance', 'priority': 7, 'min_age': 21},
    'insurance_construction': {'category': 'insurance', 'priority': 6, 'min_age': 25},
    'insurance_life': {'category': 'insurance', 'priority': 8, 'min_age': 18},
    'insurance_credit': {'category': 'insurance', 'priority': 7, 'min_age': 21},
    'insurance_mortgage': {'category': 'insurance', 'priority': 8, 'min_age': 21},
    'insurance_legal': {'category': 'insurance', 'priority': 6, 'min_age': 25},
    'insurance_deposit': {'category': 'insurance', 'priority': 5, 'min_age': 30},
    'insurance_card': {'category': 'insurance', 'priority': 6, 'min_age': 18},
    'insurance_emergency': {'category': 'insurance', 'priority': 7, 'min_age': 18},
    'insurance_drive': {'category': 'insurance', 'priority': 8, 'min_age': 18},
    'insurance_health': {'category': 'insurance', 'priority': 7, 'min_age': 18},
    'insurance_property': {'category': 'insurance', 'priority': 7, 'min_age': 25},
    'insurance_travel': {'category': 'insurance', 'priority': 7, 'min_age': 18},
}


class MultiProductRecommender:
    """
    Рекомендательная система с поддержкой ВСЕХ 70+ продуктов банка
    """
    
    def __init__(self):
        self.product_catalog = FULL_PRODUCT_CATALOG
        self.all_products = list(FULL_PRODUCT_CATALOG.keys())
        self.models = {}  # Отдельная модель для каждой категории
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def _prepare_features(self, features_df):
        """
        Подготовка фичей: удаление нечисловых колонок и заполнение пропусков
        """
        print("🔧 Подготавливаем фичи...")
        
        # Создаем копию чтобы не менять оригинал
        df = features_df.copy()
        
        # Удаляем явно нечисловые колонки
        non_numeric_cols = ['user_id', 'target_product']
        df = df.drop(non_numeric_cols, axis=1, errors='ignore')
        
        # Преобразуем все колонки в числовой формат, где возможно
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                print(f"⚠️  Не удалось преобразовать колонку {col} в числовой формат")
                df = df.drop(col, axis=1)
        
        # Заполняем пропуски
        df = df.fillna(0)
        
        # Проверяем результат
        print(f"✅ Осталось {df.shape[1]} числовых колонок")
        
        return df
        
    def create_smart_targets(self, features_df):
        """
        УЛУЧШЕННАЯ версия - максимальная персонализация рекомендаций
        """
        print("🎯 Создаем УМНЫЕ таргеты для всех продуктов...")
        
        targets = []
        
        for idx, row in features_df.iterrows():
            user_products = []
            
            # Основные метрики пользователя
            market_events = row.get('market_events', 0)
            engagement_ratio = row.get('engagement_ratio', 0)
            offers_engagement = row.get('offers_engagement', 0)
            tech_ratio = row.get('tech_interest_ratio', 0)
            home_ratio = row.get('home_interest_ratio', 0)
            sports_ratio = row.get('sports_interest_ratio', 0)
            diversity_ratio = row.get('diversity_ratio', 0)
            retail_events = row.get('retail_events', 0)
            
            # Определяем тип пользователя
            user_type = self._detect_user_type(row)
            
            # === ВКЛАДЫ ===
            if user_type in ['conservative', 'senior', 'family']:
                user_products.extend(['deposit_savings', 'deposit_pension'])
            if user_type == 'saver' or engagement_ratio < 0.1:
                user_products.extend(['savings_free', 'deposit_profitable'])
            if market_events > 100 and engagement_ratio > 0.15:
                user_products.append('deposit_special')
                    
            # === ПРЕМИУМ (только для VIP) ===
            if user_type == 'vip':
                user_products.extend(['premium_card', 'premium_package', 'premium_investment'])
            elif market_events > 150 and tech_ratio > 0.6:
                user_products.append('premium_investment')
            elif market_events > 120 and engagement_ratio > 0.2:
                user_products.append('premium_card')
                    
            # === КАРТЫ (персонализированные) ===
            # Базовые карты
            user_products.append('card_cashback')
            
            if user_type == 'digital':
                user_products.extend(['credit_card_180', 'card_psb_iz', 'card_strong_people'])
            if user_type == 'sports':
                user_products.extend(['sports_card', 'card_sportmaster', 'card_spartak', 'card_cska'])
            if user_type in ['senior', 'conservative']:
                user_products.append('pension_card')
            if market_events > 50:
                user_products.extend(['salary_card_pro', 'card_salary_plus'])
            if tech_ratio > 0.4:
                user_products.append('card_mvideo')
            if retail_events > 50:
                user_products.extend(['card_lenta', 'card_vkusvill', 'card_post_market', 'card_new_world'])
            if market_events > 30:
                user_products.append('card_resident')
                    
            # === КРЕДИТЫ (по потребностям) ===
            if user_type == 'family':
                user_products.extend(['mortgage_family', 'mortgage_new', 'mortgage_secondary'])
            if user_type == 'business':
                user_products.extend(['consumer_loan', 'refinancing'])
            if home_ratio > 0.5:
                user_products.extend(['mortgage_military', 'mortgage_far_east'])
            if market_events > 80 and offers_engagement > 10:
                user_products.extend(['mortgage_alternative', 'mortgage_castle', 'mortgage_easy'])
            if offers_engagement > 15:
                user_products.append('refinancing')
                    
            # === ИНВЕСТИЦИИ (по профилю риска) ===
            if user_type == 'investor':
                user_products.extend(['investment_stocks', 'investment_mixed', 'investment_opportunities'])
            if user_type in ['conservative', 'senior']:
                user_products.extend(['investment_bonds', 'investment_cushion', 'investment_defense'])
            if tech_ratio > 0.5:
                user_products.extend(['investment_perspective', 'investment_flow'])
            if home_ratio > 0.4:
                user_products.append('investment_resources')
            if diversity_ratio > 0.3:
                user_products.extend(['investment_world', 'investment_east'])
            if market_events > 100:
                user_products.extend(['investment_dividend', 'investment_stocks'])
                    
            # === СТРАХОВАНИЕ (по образу жизни) ===
            # Базовые страховки
            user_products.append('insurance_life')
            
            if market_events > 20:
                user_products.extend(['insurance_osago', 'insurance_card'])
            if user_type == 'family':
                user_products.extend(['insurance_property', 'insurance_mortgage', 'insurance_emergency'])
            if user_type == 'sports':
                user_products.extend(['insurance_health', 'insurance_drive', 'insurance_emergency'])
            if user_type == 'business':
                user_products.extend(['insurance_credit', 'insurance_legal', 'insurance_job_loss'])
            if home_ratio > 0.6:
                user_products.extend(['insurance_property', 'insurance_construction'])
            if sports_ratio > 0.4:
                user_products.append('insurance_drive')
            if diversity_ratio > 0.4:
                user_products.append('insurance_travel')
            if market_events > 60:
                user_products.append('insurance_deposit')
            
            # Убираем дубликаты и ограничиваем разумным количеством
            user_products = list(set(user_products))
            targets.append(user_products)
        
        return targets

    def _detect_user_type(self, user_data):
        """
        Определяем тип пользователя для максимальной персонализации
        """
        market_events = user_data.get('market_events', 0)
        engagement_ratio = user_data.get('engagement_ratio', 0)
        tech_ratio = user_data.get('tech_interest_ratio', 0)
        home_ratio = user_data.get('home_interest_ratio', 0)
        sports_ratio = user_data.get('sports_interest_ratio', 0)
        offers_engagement = user_data.get('offers_engagement', 0)
        
        # Логика определения типа
        if market_events > 200 and engagement_ratio > 0.2 and tech_ratio > 0.6:
            return 'vip'
        elif market_events > 150 and tech_ratio > 0.5:
            return 'digital'
        elif market_events > 100 and offers_engagement > 15:
            return 'investor'
        elif home_ratio > 0.7:
            return 'family'
        elif sports_ratio > 0.6:
            return 'sports'
        elif market_events > 120 and engagement_ratio > 0.15:
            return 'business'
        elif market_events < 30 or home_ratio > 0.8:
            return 'senior'
        elif engagement_ratio < 0.08:
            return 'conservative'
        elif market_events > 80:
            return 'active'
        else:
            return 'casual'
        
    def train(self, features_df):
        """
        Обучаем модель с multi-label подходом
        """
        print(f"🤖 ОБУЧАЕМ СИСТЕМУ ДЛЯ {len(self.all_products)} ПРОДУКТОВ...")
        
        # 1. Подготовка фичей
        X = self._prepare_features(features_df)
        self.feature_names = X.columns.tolist()
        
        print(f"📋 Используется {len(self.feature_names)} числовых признаков")
        
        # 2. Создаем таргеты
        targets = self.create_smart_targets(features_df)
        
        # Статистика покрытия
        all_recommended = [p for sublist in targets for p in sublist]
        unique_products = set(all_recommended)
        print(f"📊 Покрытие продуктов: {len(unique_products)}/{len(self.all_products)}")
        print(f"📈 Среднее кол-во продуктов на пользователя: {len(all_recommended)/len(targets):.1f}")
        
        # 3. Масштабирование
        print("⚖️  Масштабируем фичи...")
        X_scaled = self.scaler.fit_transform(X)
        
        # 4. Multi-label кодирование
        print("🔢 Кодируем таргеты...")
        mlb = MultiLabelBinarizer(classes=self.all_products)
        y_binary = mlb.fit_transform(targets)
        
        # 5. Разделение
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_binary, test_size=0.2, random_state=42
        )
        
        # 6. Обучаем отдельную модель для каждого продукта (One-vs-Rest)
        print("🚀 Обучаем модели...")
        
        trained_models_count = 0
        for i, product in enumerate(self.all_products):
            if i % 10 == 0:
                print(f"   Прогресс: {i}/{len(self.all_products)}")
            
            # Пропускаем продукты без примеров или с одним классом
            positive_examples = y_train[:, i].sum()
            negative_examples = len(y_train) - positive_examples
            
            # УМЕНЬШИЛИ ТРЕБОВАНИЯ: минимум 2 примера положительного класса
            if positive_examples < 2:
                print(f"   ⏭️  Пропускаем {product}: недостаточно данных ({positive_examples}+ примеров)")
                continue
                
            try:
                model = xgb.XGBClassifier(
                    n_estimators=50,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=42,
                    verbosity=0,
                    scale_pos_weight=negative_examples / (positive_examples + 1)
                )
                
                model.fit(X_train, y_train[:, i])
                self.models[product] = model
                trained_models_count += 1
                print(f"   ✅ Обучили {product}: {positive_examples}+ примеров")
                
            except Exception as e:
                print(f"   ❌ Ошибка при обучении {product}: {e}")
        
        print(f"✅ Обучено {trained_models_count} моделей из {len(self.all_products)}")
        
        # 7. Оценка
        if trained_models_count > 0:
            self._evaluate(X_test, y_test, mlb)
        else:
            print("❌ Не обучено ни одной модели!")
            return self
        
        # 8. Сохранение
        self._save_models()
        
        return self

    def _save_models(self):
        """Сохранение моделей"""
        import os
        os.makedirs('models', exist_ok=True)
        
        # Сохраняем каждую модель
        for product, model in self.models.items():
            model.save_model(f'models/model_{product}.json')
        
        # Метаданные
        with open('models/recommender_meta.pkl', 'wb') as f:
            pickle.dump({
                'all_products': self.all_products,
                'feature_names': self.feature_names,
                'product_catalog': self.product_catalog,
                'trained_models': list(self.models.keys())
            }, f)
        
        # Scaler
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"💾 Модели сохранены в папку models/ ({len(self.models)} моделей)")
    
    def _evaluate(self, X_test, y_test, mlb):
        """Оценка качества"""
        print("\n📊 ОЦЕНКА КАЧЕСТВА:")
        
        # Предсказания
        y_pred = np.zeros_like(y_test)
        for i, product in enumerate(self.all_products):
            if product in self.models:
                y_pred[:, i] = self.models[product].predict(X_test)
        
        # Метрики
        hamming = hamming_loss(y_test, y_pred)
        jaccard = jaccard_score(y_test, y_pred, average='samples', zero_division=1)
        
        print(f"   Hamming Loss: {hamming:.4f}")
        print(f"   Jaccard Score: {jaccard:.4f}")
        
        # Покрытие
        test_coverage = (y_pred.sum(axis=1) > 0).mean()
        print(f"   Покрытие тестовой выборки: {test_coverage:.1%}")
        
        # Топ продуктов
        product_predictions = y_pred.sum(axis=0)
        top_products = sorted(
            zip(self.all_products, product_predictions), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        print("\n🔝 Топ-10 рекомендуемых продуктов:")
        for product, count in top_products:
            print(f"   {product}: {int(count)} раз")
    
    def recommend(self, user_features, top_n=10, category_filter=None):
        """
        УЛУЧШЕННЫЕ рекомендации с максимальной персонализацией
        """
        if not self.models:
            print("❌ Модели не обучены!")
            return []
        
        # Подготовка данных
        user_features_clean = {k: v for k, v in user_features.items() if k in self.feature_names}
        X = pd.DataFrame([user_features_clean])[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Определяем тип пользователя для бустинга
        user_type = self._detect_user_type(user_features)
        
        # Предсказания вероятностей с учетом типа пользователя
        scores = {}
        for product, model in self.models.items():
            if category_filter and self.product_catalog[product]['category'] != category_filter:
                continue
            
            try:
                proba = model.predict_proba(X_scaled)[0, 1]
                
                # УМНЫЙ БУСТИНГ на основе типа пользователя и приоритета
                priority = self.product_catalog[product]['priority']
                category = self.product_catalog[product]['category']
                
                # Базовая оценка
                base_score = proba * (priority / 10.0)
                
                # Бустинг по типу пользователя
                type_boost = self._get_type_boost(user_type, category, product)
                
                # Бустинг по поведению
                behavior_boost = self._get_behavior_boost(user_features, category)
                
                # Финальная оценка
                final_score = base_score * type_boost * behavior_boost
                
                scores[product] = {
                    'score': final_score,
                    'probability': proba,
                    'category': category,
                    'priority': priority,
                    'type_boost': type_boost,
                    'behavior_boost': behavior_boost
                }
            except Exception as e:
                continue
        
        # Ранжирование
        sorted_products = sorted(
            scores.items(), 
            key=lambda x: x[1]['score'], 
            reverse=True
        )[:top_n]
        
        # Форматирование
        recommendations = []
        for product_id, data in sorted_products:
            recommendations.append({
                'product_id': product_id,
                'category': data['category'],
                'score': f"{data['score']:.3f}",
                'probability': f"{data['probability']:.1%}",
                'priority': data['priority'],
                'explanation': self._generate_detailed_explanation(user_features, product_id, user_type)
            })
        
        return recommendations

    def _get_type_boost(self, user_type, category, product):
        """
        Бустинг на основе типа пользователя
        """
        type_boosts = {
            'vip': {'premium': 2.0, 'investments': 1.5, 'cards': 1.3},
            'digital': {'cards': 1.8, 'investments': 1.6, 'premium': 1.4},
            'investor': {'investments': 2.0, 'premium': 1.5},
            'family': {'loans': 1.8, 'insurance': 1.6, 'savings': 1.4},
            'sports': {'cards': 1.7, 'insurance': 1.5},
            'business': {'loans': 1.8, 'premium': 1.6, 'investments': 1.4},
            'senior': {'savings': 1.8, 'cards': 1.6, 'insurance': 1.4},
            'conservative': {'savings': 1.7, 'insurance': 1.3}
        }
        
        boost = 1.0
        if user_type in type_boosts:
            boosts = type_boosts[user_type]
            if category in boosts:
                boost = boosts[category]
        
        return boost

    def _get_behavior_boost(self, user_features, category):
        """
        Бустинг на основе поведения пользователя
        """
        boost = 1.0
        
        market_events = user_features.get('market_events', 0)
        engagement_ratio = user_features.get('engagement_ratio', 0)
        tech_ratio = user_features.get('tech_interest_ratio', 0)
        
        if category == 'premium' and market_events > 150:
            boost *= 1.5
        if category == 'investments' and tech_ratio > 0.6:
            boost *= 1.4
        if category == 'cards' and engagement_ratio > 0.15:
            boost *= 1.3
        
        return boost

    def _generate_detailed_explanation(self, user_features, product_id, user_type):
        """
        Детальное объяснение рекомендации
        """
        reasons = []
        
        category = self.product_catalog[product_id]['category']
        market_events = user_features.get('market_events', 0)
        engagement_ratio = user_features.get('engagement_ratio', 0)
        tech_ratio = user_features.get('tech_interest_ratio', 0)
        home_ratio = user_features.get('home_interest_ratio', 0)
        
        # Общие причины
        if market_events > 100:
            reasons.append("высокая активность")
        elif market_events < 30:
            reasons.append("стабильное поведение")
        
        # Причины по типу пользователя
        if user_type == 'vip':
            reasons.append("VIP-статус")
        elif user_type == 'digital':
            reasons.append("технологическая вовлеченность")
        elif user_type == 'family':
            reasons.append("семейный профиль")
        
        # Причины по категории продукта
        if category == 'premium' and engagement_ratio > 0.15:
            reasons.append("высокая лояльность")
        if category == 'investments' and tech_ratio > 0.5:
            reasons.append("интерес к инновациям")
        if category == 'loans' and home_ratio > 0.6:
            reasons.append("потребность в финансировании")
        
        if not reasons:
            reasons.append("идеально подходит вашему профилю")
        
        return ", ".join(reasons)


# ===================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ===================
if __name__ == "__main__":
    import os
    
    # 1. Загружаем данные
    print("📥 Загружаем фичи...")
    try:
        # Пробуем разные возможные файлы
        if os.path.exists('user_features_enhanced.pq'):
            features_df = pd.read_parquet('user_features_enhanced.pq')
        elif os.path.exists('train_features.pq'):
            features_df = pd.read_parquet('train_features.pq')
        elif os.path.exists('test_features.pq'):
            features_df = pd.read_parquet('test_features.pq')
        else:
            print("❌ Файлы с фичами не найдены!")
            exit(1)
            
        print(f"📊 Размер данных: {features_df.shape}")
        print(f"📋 Колонки: {features_df.columns.tolist()}")
        
    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        exit(1)
    
    # 2. Обучаем систему
    recommender = MultiProductRecommender()
    recommender.train(features_df)
    
    # 3. Тестовые рекомендации
    if recommender.models:
        print("\n" + "="*60)
        print("🎯 ПРИМЕРЫ РЕКОМЕНДАЦИЙ")
        print("="*60)
        
        for i in range(min(3, len(features_df))):
            user = features_df.iloc[i].to_dict()
            print(f"\n👤 Пользователь {i+1}:")
            print(f"   Активность: {user.get('market_events', 0):.0f} событий")
            
            recs = recommender.recommend(user, top_n=5)
            
            if recs:
                print(f"   📋 Топ-5 рекомендаций:")
                for j, rec in enumerate(recs, 1):
                    print(f"      {j}. {rec['product_id']:30} | {rec['category']:15} | {rec['probability']:6} | {rec['explanation']}")
            else:
                print("   ❌ Нет рекомендаций")
        
        print("\n✅ ГОТОВО! Система рекомендует продукты банка")
    else:
        print("❌ Система не обучена, рекомендации невозможны")