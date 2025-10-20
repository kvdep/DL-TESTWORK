# Source code for data_preprocessing from PLDataModule

    def data_preprocessing(self, data: pd.DataFrame, y_train: Optional[pd.Series] = None, fit_mode: bool = False) -> np.ndarray:
        """
        Финальная версия препроцессинга, объединяющая лучшие идеи из обоих подходов.
        """
        df = data.copy()

        # --- ШАГ 1: Начальная очистка данных (из старого кода) ---

        # 1.1. Восстановление пропущенных location по станции метро
        if fit_mode: # Эту информацию нужно создать только один раз
            self.metro_dict = {
                "Москва": {"Авиамоторная", "Автозаводская", "Академическая", "Александровский сад", "Алексеевская", "Алма-Атинская", "Алтуфьево", "Аннино", "Арбатская", "Аэропорт", "Бабушкинская", "Багратионовская", "Баррикадная", "Бауманская", "Беговая", "Беломорская", "Белорусская", "Беляево", "Бибирево", "Библиотека имени Ленина", "Битцевский парк", "Борисово", "Боровицкая", "Боровское шоссе", "Ботанический сад", "Братиславская", "Бульвар адмирала Ушакова", "Бульвар Дмитрия Донского", "Бульвар Рокоссовского", "Бунинская аллея", "Бутырская", "Варшавская", "ВДНХ", "Верхние Лихоборы", "Владыкино", "Водный стадион", "Войковская", "Волгоградский проспект", "Волжская", "Волоколамская", "Воробьевы горы", "Выставочная", "Выхино", "Говорово", "Деловой центр", "Динамо", "Дмитровская", "Добрынинская", "Домодедовская", "Достоевская", "Дубровка", "Жулебино", "Зябликово", "Измайловская", "Калужская", "Кантемировская", "Каховская", "Каширская", "Киевская", "Китай-город", "Кожуховская", "Коломенская", "Коммунарка", "Комсомольская", "Коньково", "Косино", "Котельники", "Красногвардейская", "Краснопресненская", "Красносельская", "Красные ворота", "Крестьянская застава", "Кропоткинская", "Крылатское", "Кузнецкий мост", "Кузьминки", "Кунцевская", "Курская", "Кутузовская", "Ленинский проспект", "Лермонтовский проспект", "Лесопарковая", "Лефортово", "Ломоносовский проспект", "Лубянка", "Лухмановская", "Люблино", "Марксистская", "Марьина роща", "Марьино", "Маяковская", "Медведково", "Международная", "Менделеевская", "Минская", "Митино", "Мичуринский проспект", "Молодежная", "Мякинино", "Нагатинская", "Нагорная", "Нахимовский проспект", "Некрасовка", "Нижегородская", "Новогиреево", "Новокосино", "Новокузнецкая", "Новопеределкино", "Новослободская", "Новоясеневская", "Новые Черемушки", "Озерная", "Окружная", "Окская", "Октябрьская", "Октябрьское поле", "Ольховая", "Орехово", "Отрадное", "Охотный ряд", "Павелецкая", "Парк культуры", "Парк Победы", "Партизанская", "Первомайская", "Перово", "Петровский парк", "Петровско-Разумовская", "Печатники", "Пионерская", "Планерная", "Площадь Ильича", "Площадь Революции", "Полежаевская", "Полянка", "Пражская", "Преображенская площадь", "Прокшино", "Пролетарская", "Проспект Вернадского", "Проспект Мира", "Профсоюзная", "Пушкинская", "Пятницкое шоссе", "Раменки", "Рассказовка", "Речной вокзал", "Рижская", "Римская", "Румянцево", "Рязанский проспект", "Савеловская", "Саларьево", "Свиблово", "Севастопольская", "Селигерская", "Семеновская", "Серпуховская", "Славянский бульвар", "Смоленская", "Сокол", "Сокольники", "Солнцево", "Спартак", "Спортивная", "Сретенский бульвар", "Стахановская", "Строгино", "Студенческая", "Сухаревская", "Сходненская", "Таганская", "Тверская", "Театральная", "Текстильщики", "Теплый Стан", "Технопарк", "Тимирязевская", "Третьяковская", "Тропарево", "Трубная", "Тульская", "Тургеневская", "Тушинская", "Улица 1905 года", "Улица академика Янгеля", "Улица Горчакова", "Улица Дмитриевского", "Улица Скобелевская", "Улица Старокачаловская", "Университет", "Филатов луг", "Филевский парк", "Фили", "Фонвизинская", "Фрунзенская", "Ховрино", "Хорошевская", "Царицыно", "Цветной бульвар", "ЦСКА", "Черкизовская", "Чертановская", "Чеховская", "Чистые пруды", "Чкаловская", "Шаболовская", "Шелепиха", "Шипиловская", "Шоссе Энтузиастов", "Щелковская", "Щукинская", "Электрозаводская", "Юго-Восточная", "Юго-Западная", "Южная", "Ясенево"},
                "Казань": {"Авиастроительная", "Аметьево", "Горки", "Козья слобода", "Кремлёвская", "Площадь Габдуллы Тукая", "Проспект Победы", "Северный вокзал", "Суконная слобода", "Яшьлек", "Юность"}
            }
        
        missing_loc_idx = df[df.location.isna()].index
        for idx in missing_loc_idx:
            station = df.loc[idx, "underground"]
            if station in self.metro_dict["Москва"]:
                df.loc[idx, "location"] = "Москва"
            elif station in self.metro_dict["Казань"]:
                df.loc[idx, "location"] = "Казань"

        # 1.2. Перекрестное заполнение этажей
        floor_na_mask = df["floor"].isna()
        floors_count_na_mask = df["floors_count"].isna()
        df.loc[floor_na_mask, "floor"] = df.loc[floor_na_mask, "floors_count"]
        df.loc[floors_count_na_mask, "floors_count"] = df.loc[floors_count_na_mask, "floor"]

        # --- ШАГ 2: Продвинутый Feature Engineering ---
        # (Объединяем наши предыдущие наработки)
        df['is_studio'] = (df['rooms_count'] <= 0).astype(int)
        df['rooms_count'] = df['rooms_count'].apply(lambda x: 1 if x <= 0 else x)

        if fit_mode:
            if y_train is None: raise ValueError("y_train is required for fit_mode.")
            full_train_df = pd.concat([df, y_train], axis=1)
            median_price_by_loc = full_train_df.groupby('location')[self.y_label].median().sort_values()
            self.location_rank_map = {city: rank for rank, city in enumerate(median_price_by_loc.index)}
            self.median_rank = df['location'].map(self.location_rank_map).median()

        df['location_rank'] = df['location'].map(self.location_rank_map).fillna(self.median_rank)
        
        # --- ШАГ 3: Умное заполнение total_meters (из старого кода) ---
        if fit_mode:
            # Создаем карту {кол-во комнат: медианная площадь}
            self.rooms_to_meters_map = df.groupby('rooms_count')['total_meters'].median()
        
        # Применяем карту для заполнения пропусков
        missing_meters_mask = df['total_meters'].isna()
        fill_values = df.loc[missing_meters_mask, 'rooms_count'].map(self.rooms_to_meters_map)
        df.loc[missing_meters_mask, 'total_meters'] = fill_values

        # --- ШАГ 4: Генерация остальных признаков ---
        df['floor_ratio'] = (df['floor'] / (df['floors_count'] + 1e-6)).astype(float)
        df['meters_per_room'] = (df['total_meters'] / (df['rooms_count'].replace(0, 1) + 1e-6)).astype(float)
        
        # Удаляем столбцы, которые больше не нужны или будут закодированы
        cols_to_drop = ['author', 'author_type', 'deal_type', 'accommodation_type', 'commissions', 
                        'district', 'street', 'house_number', 'underground', 'location', 'ID']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

        # --- ШАГ 5: Финальная обработка (импьютеры-страховка, кодирование, масштабирование) ---
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

        if fit_mode:
            # Импьютеры теперь служат для "подчистки" оставшихся пропусков
            self.numerical_imputer = SimpleImputer(strategy='median')
            self.categorical_imputer = SimpleImputer(strategy='most_frequent')
            df[numerical_cols] = self.numerical_imputer.fit_transform(df[numerical_cols])
            if categorical_cols:
                df[categorical_cols] = self.categorical_imputer.fit_transform(df[categorical_cols])
        else:
            if not all([self.numerical_imputer, self.categorical_imputer]): raise RuntimeError("Imputers have not been fitted.")
            df[numerical_cols] = self.numerical_imputer.transform(df[numerical_cols])
            if categorical_cols:
                df[categorical_cols] = self.categorical_imputer.transform(df[categorical_cols])

        df = pd.get_dummies(df, dummy_na=True, dtype=int)

        if fit_mode:
            self.final_columns = df.columns.tolist()
            self.scaler = StandardScaler()
            scaled_data = self.scaler.fit_transform(df)
        else:
            df = df.reindex(columns=self.final_columns, fill_value=0)
            scaled_data = self.scaler.transform(df)

        return scaled_data.astype(float)


# Source code for save_preprocessing from PLDataModule

    def save_preprocessing(self, directory_path: str) -> Optional[str]:
        """
        Сохраняет ВСЕ артефакты, необходимые для воспроизведения предобработки.
        """
        self.preprocessing_artifacts = {
            'scaler': self.scaler,
            'numerical_imputer': self.numerical_imputer,
            'categorical_imputer': self.categorical_imputer,
            'final_columns': self.final_columns,
            'top_k_values': self.top_k_values,
            'top_k_config': self.top_k_config,
            # НОВЫЕ АРТЕФАКТЫ
            'metro_dict': self.metro_dict,
            'location_rank_map': self.location_rank_map,
            'median_rank': self.median_rank,
            'rooms_to_meters_map': self.rooms_to_meters_map
        }

        # Проверка, что все артефакты созданы
        for key, value in self.preprocessing_artifacts.items():
            if value is None or (isinstance(value, (dict, list)) and not value):
                # Игнорируем пустые top_k_config/values, если они не заданы
                if key in ['top_k_config', 'top_k_values']:
                    continue
                print(f"Предупреждение: Артефакт '{key}' не был создан. Сохранение отменено.")
                return None

        preprocess_dir = os.path.join(directory_path, "preprocessing")
        os.makedirs(preprocess_dir, exist_ok=True)
        artifacts_path = os.path.join(preprocess_dir, "preprocessing_artifacts.joblib")

        try:
            joblib.dump(self.preprocessing_artifacts, artifacts_path)
            print(f"Артефакты предобработки успешно сохранены в: {artifacts_path}")
            return preprocess_dir
        except Exception as e:
            print(f"Ошибка при сохранении артефактов предобработки: {e}")
            return None
