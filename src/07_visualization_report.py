# src/07_visualization_report.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pickle

plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

class RecommenderVisualizer:
    """
    Визуализация результатов рекомендательной системы
    """
    
    def __init__(self, features_df, models_dir='models'):
        self.features_df = features_df
        
        # Загружаем метаданные
        with open(f'{models_dir}/recommender_meta.pkl', 'rb') as f:
            meta = pickle.load(f)
            self.product_catalog = meta['product_catalog']
    
    def plot_product_coverage(self, recommendations_list):
        """
        График покрытия продуктов
        """
        # Собираем все рекомендации
        all_products = [p for recs in recommendations_list for p, _ in recs]
        product_counts = Counter(all_products)
        
        # Сортируем
        products_sorted = sorted(product_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Рисуем
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # График 1: Топ-30 продуктов
        top_products = products_sorted[:30]
        names = [p[0][:25] for p in top_products]
        counts = [p[1] for p in top_products]
        
        bars = ax1.barh(range(len(names)), counts, color='steelblue')
        ax1.set_yticks(range(len(names)))
        ax1.set_yticklabels(names, fontsize=8)
        ax1.set_xlabel('Количество рекомендаций', fontsize=11)
        ax1.set_title('📊 Топ-30 рекомендуемых продуктов', fontsize=13, fontweight='bold')
        ax1.invert_yaxis()
        
        # Добавляем значения
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                    f'{int(width)}', ha='left', va='center', fontsize=8)
        
        # График 2: Распределение по категориям
        category_counts = {}
        for product, count in product_counts.items():
            cat = self.product_catalog[product]['category']
            category_counts[cat] = category_counts.get(cat, 0) + count
        
        categories = list(category_counts.keys())
        cat_counts = list(category_counts.values())
        
        colors = plt.cm.Set3(range(len(categories)))
        wedges, texts, autotexts = ax2.pie(
            cat_counts, 
            labels=categories,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors
        )
        
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_fontweight('bold')
        
        ax2.set_title('📈 Распределение по категориям', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('product_coverage.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("💾 Сохранено: product_coverage.png")
    
    def plot_user_segments(self):
        """
        Сегментация пользователей
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Распределение активности
        ax1 = axes[0, 0]
        activity_levels = pd.cut(
            self.features_df['market_events'],
            bins=[0, 30, 80, 150, 1000],
            labels=['Низкая', 'Средняя', 'Высокая', 'Очень высокая']
        )
        activity_counts = activity_levels.value_counts()
        
        bars = ax1.bar(activity_counts.index, activity_counts.values, color='coral')
        ax1.set_xlabel('Уровень активности', fontsize=11)
        ax1.set_ylabel('Количество пользователей', fontsize=11)
        ax1.set_title('👥 Распределение по активности', fontsize=12, fontweight='bold')
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10)
        
        # 2. Интересы пользователей
        ax2 = axes[0, 1]
        interests = pd.DataFrame({
            'Технологии': self.features_df['tech_interest_ratio'],
            'Спорт': self.features_df['sports_interest_ratio'],
            'Недвижимость': self.features_df['home_interest_ratio']
        })
        
        interests.boxplot(ax=ax2, patch_artist=True)
        ax2.set_ylabel('Уровень интереса', fontsize=11)
        ax2.set_title('🎯 Распределение интересов', fontsize=12, fontweight='bold')
        ax2.set_xticklabels(interests.columns, rotation=15, ha='right')
        
        # 3. Вовлеченность
        ax3 = axes[1, 0]
        engagement_data = [
            self.features_df['engagement_ratio'].dropna(),
            self.features_df['offers_engagement_ratio'].dropna(),
            self.features_df['retail_purchase_intent'].dropna()
        ]
        
        bp = ax3.boxplot(engagement_data, labels=['Маркетплейс', 'Офферы', 'Покупки'],
                        patch_artist=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        ax3.set_ylabel('Коэффициент вовлеченности', fontsize=11)
        ax3.set_title('💡 Вовлеченность пользователей', fontsize=12, fontweight='bold')
        ax3.set_xticklabels(['Маркетплейс', 'Офферы', 'Покупки'], rotation=15, ha='right')
        
        # 4. Корреляция активности и вовлеченности
        ax4 = axes[1, 1]
        
        # Фильтруем выбросы
        data = self.features_df[
            (self.features_df['market_events'] < 200) &
            (self.features_df['offers_engagement'] < 20)
        ]
        
        scatter = ax4.scatter(
            data['market_events'],
            data['offers_engagement'],
            c=data['engagement_ratio'],
            cmap='viridis',
            alpha=0.6,
            s=30
        )
        
        ax4.set_xlabel('Активность в маркетплейсе', fontsize=11)
        ax4.set_ylabel('Вовлеченность в офферы', fontsize=11)
        ax4.set_title('🔗 Корреляция активности', fontsize=12, fontweight='bold')
        
        plt.colorbar(scatter, ax=ax4, label='Engagement Ratio')
        
        plt.tight_layout()
        plt.savefig('user_segments.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("💾 Сохранено: user_segments.png")
    
    def plot_recommendation_quality(self, recommendations_list):
        """
        Качество рекомендаций
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Распределение количества рекомендаций на пользователя
        ax1 = axes[0, 0]
        rec_counts = [len(recs) for recs in recommendations_list]
        
        ax1.hist(rec_counts, bins=range(1, 12), color='skyblue', edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Количество рекомендаций', fontsize=11)
        ax1.set_ylabel('Количество пользователей', fontsize=11)
        ax1.set_title('📊 Распределение рекомендаций', fontsize=12, fontweight='bold')
        ax1.axvline(np.mean(rec_counts), color='red', linestyle='--', 
                   label=f'Среднее: {np.mean(rec_counts):.1f}')
        ax1.legend()
        
        # 2. Распределение скоров
        ax2 = axes[0, 1]
        all_scores = [score for recs in recommendations_list for _, data in recs for score in [data['score']]]
        
        ax2.hist(all_scores, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Скор рекомендации', fontsize=11)
        ax2.set_ylabel('Частота', fontsize=11)
        ax2.set_title('🎯 Распределение скоров', fontsize=12, fontweight='bold')
        ax2.axvline(np.median(all_scores), color='green', linestyle='--',
                   label=f'Медиана: {np.median(all_scores):.2f}')
        ax2.legend()
        
        # 3. Топ категорий по скорам
        ax3 = axes[1, 0]
        category_scores = {}
        for recs in recommendations_list:
            for product, data in recs:
                cat = data['category']
                if cat not in category_scores:
                    category_scores[cat] = []
                category_scores[cat].append(data['score'])
        
        categories = list(category_scores.keys())
        avg_scores = [np.mean(category_scores[cat]) for cat in categories]
        
        bars = ax3.barh(categories, avg_scores, color='mediumseagreen')
        ax3.set_xlabel('Средний скор', fontsize=11)
        ax3.set_title('📈 Средний скор по категориям', fontsize=12, fontweight='bold')
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax3.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{width:.2f}',
                    ha='left', va='center', fontsize=9)
        
        # 4. Diversity Score
        ax4 = axes[1, 1]
        user_diversity = []
        
        for recs in recommendations_list:
            categories_in_recs = set(data['category'] for _, data in recs)
            diversity = len(categories_in_recs) / max(1, len(recs))
            user_diversity.append(diversity)
        
        ax4.hist(user_diversity, bins=20, color='plum', edgecolor='black', alpha=0.7)
        ax4.set_xlabel('Diversity Score', fontsize=11)
        ax4.set_ylabel('Количество пользователей', fontsize=11)
        ax4.set_title('🌈 Разнообразие рекомендаций', fontsize=12, fontweight='bold')
        ax4.axvline(np.mean(user_diversity), color='darkviolet', linestyle='--',
                   label=f'Среднее: {np.mean(user_diversity):.2f}')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('recommendation_quality.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("💾 Сохранено: recommendation_quality.png")
    
    def generate_summary_report(self, recommendations_list):
        """
        Генерируем итоговый отчет
        """
        print("\n" + "="*80)
        print("📋 ИТОГОВЫЙ ОТЧЕТ ПО РЕКОМЕНДАТЕЛЬНОЙ СИСТЕМЕ")
        print("="*80)
        
        # 1. Общая статистика
        total_users = len(self.features_df)
        total_recs = sum(len(recs) for recs in recommendations_list)
        avg_recs = total_recs / total_users
        
        print(f"\n📊 ОБЩАЯ СТАТИСТИКА:")
        print(f"   Пользователей проанализировано: {total_users:,}")
        print(f"   Всего рекомендаций выдано: {total_recs:,}")
        print(f"   Среднее на пользователя: {avg_recs:.2f}")
        
        # 2. Покрытие продуктов
        all_products = [p for recs in recommendations_list for p, _ in recs]
        unique_products = set(all_products)
        total_available = len(self.product_catalog)
        
        print(f"\n🎯 ПОКРЫТИЕ ПРОДУКТОВ:")
        print(f"   Уникальных продуктов рекомендовано: {len(unique_products)}/{total_available}")
        print(f"   Процент покрытия: {len(unique_products)/total_available*100:.1f}%")
        
        # 3. Топ продуктов
        product_counts = Counter(all_products)
        top_10 = product_counts.most_common(10)
        
        print(f"\n🔝 ТОП-10 РЕКОМЕНДУЕМЫХ ПРОДУКТОВ:")
        for i, (product, count) in enumerate(top_10, 1):
            pct = count / total_recs * 100
            print(f"   {i:2}. {product:35} : {count:5} ({pct:5.2f}%)")
        
        # 4. Распределение по категориям
        category_counts = {}
        for product, count in product_counts.items():
            cat = self.product_catalog[product]['category']
            category_counts[cat] = category_counts.get(cat, 0) + count
        
        print(f"\n📈 РАСПРЕДЕЛЕНИЕ ПО КАТЕГОРИЯМ:")
        for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            pct = count / total_recs * 100
            print(f"   {cat:20} : {count:6} ({pct:5.2f}%)")
        
        # 5. Качество
        all_scores = [score for recs in recommendations_list for _, data in recs for score in [data['score']]]
        
        print(f"\n⭐ КАЧЕСТВО РЕКОМЕНДАЦИЙ:")
        print(f"   Средний скор: {np.mean(all_scores):.3f}")
        print(f"   Медианный скор: {np.median(all_scores):.3f}")
        print(f"   Мин/Макс скор: {np.min(all_scores):.3f} / {np.max(all_scores):.3f}")
        
        # 6. Diversity
        user_diversity = []
        for recs in recommendations_list:
            categories_in_recs = set(data['category'] for _, data in recs)
            diversity = len(categories_in_recs) / max(1, len(recs))
            user_diversity.append(diversity)
        
        print(f"\n🌈 РАЗНООБРАЗИЕ:")
        print(f"   Средний Diversity Score: {np.mean(user_diversity):.3f}")
        print(f"   (1.0 = максимальное разнообразие)")
        
        # 7. Бизнес-метрики
        print(f"\n💼 БИЗНЕС-ЦЕННОСТЬ:")
        print(f"   ✅ Система покрывает {len(unique_products)/total_available*100:.0f}% каталога")
        print(f"   ✅ Средняя релевантность: {np.mean(all_scores):.1%}")
        print(f"   ✅ Разнообразие категорий: {np.mean(user_diversity):.1%}")
        
        high_quality = sum(1 for s in all_scores if s > 0.3) / len(all_scores) * 100
        print(f"   ✅ Доля высококачественных рекомендаций (>0.3): {high_quality:.1f}%")
        
        print("\n" + "="*80)


# ===================
# ЗАПУСК ВИЗУАЛИЗАЦИИ
# ===================
def main():
    print("📊 ГЕНЕРАЦИЯ ВИЗУАЛИЗАЦИЙ И ОТЧЕТОВ")
    print("="*80)
    
    # Загружаем данные
    features_df = pd.read_parquet('user_features_enhanced.pq')
    
    visualizer = RecommenderVisualizer(features_df)
    
    # Генерируем тестовые рекомендации
    print("\n🔄 Генерируем рекомендации для анализа...")
    
    from src.src.src_05_multi_product_recommender import MultiProductRecommender
    recommender = MultiProductRecommender()
    
    # Загружаем модели (упрощенно)
    with open('models/recommender_meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    recommender.all_products = meta['all_products']
    recommender.feature_names = meta['feature_names']
    recommender.product_catalog = meta['product_catalog']
    
    with open('models/scaler.pkl', 'rb') as f:
        recommender.scaler = pickle.load(f)
    
    # Загружаем модели
    import xgboost as xgb
    recommender.models = {}
    for product in recommender.all_products:
        try:
            model = xgb.XGBClassifier()
            model.load_model(f'models/model_{product}.json')
            recommender.models[product] = model
        except:
            pass
    
    # Генерируем рекомендации
    sample_size = min(500, len(features_df))
    recommendations_list = []
    
    for i in range(sample_size):
        user = features_df.iloc[i].to_dict()
        recs = recommender.recommend(user, top_n=7, min_score=0.05)
        recommendations_list.append(recs)
    
    print(f"✅ Сгенерировано рекомендаций для {sample_size} пользователей")
    
    # Визуализации
    print("\n📊 Создаем графики...")
    visualizer.plot_product_coverage(recommendations_list)
    visualizer.plot_user_segments()
    visualizer.plot_recommendation_quality(recommendations_list)
    
    # Отчет
    visualizer.generate_summary_report(recommendations_list)
    
    print("\n✅ ВСЕ ВИЗУАЛИЗАЦИИ ГОТОВЫ!")
    print("   - product_coverage.png")
    print("   - user_segments.png")
    print("   - recommendation_quality.png")


if __name__ == "__main__":
    main()