# 2. Импорт конфигов
import config as cnfg
import config_paths as cnfg_p

# 3. Проверка и создание папок (если их нет)
# При первом запуске он создаст папки data/raw, data/processed и т.д.
cnfg_p.init_structure()

# 5. Импорт ваших расчетных модулей
from scripts import import_data as idt
from scripts import data_manager as dm



def start_generate_data():
    # Ячейка 3: Запуск обработки циклов
    df_logs, df_registry = idt.run_extraction(
        raw_dir=cnfg_p.RAW_DATA_DIR,
        settings=cnfg.EXTRACTION_SETTINGS,
        col_config=cnfg.COLUMNS_CONFIG
    )
    
    print(f"Экстракция завершена. Обработано {len(df_registry)} циклов.")

    dm.save_database(df_logs, cnfg_p.LOGS_FILE)
    dm.save_database(df_registry, cnfg_p.REGISTRY_FILE)

    # Импорт данных настроек дезоксидации из техкарты
    df_reference_setpoint = dm.load_database(cnfg_p.REFERENCE_SETPOINT_FILE)
    df_reference_setpoint = df_reference_setpoint.astype({'Dia_Shell': int, 'L_Shell': int})
    print("Загружена техкарта")

    # Запуск классификации циклов по сортаменту
    df_registry = idt.classify_registry_generic(df_registry, cnfg.CLASSIFICATION_TASKS)
    print("Классификация циклов завершена")

    # Ячейка 5: Усреднение циклов по сортаменту  
    df_avg_logs, df_avg_registry = dm.get_averaged_profiles(
        df_logs, 
        df_registry, 
        cnfg.CLASSIFICATION_TASKS, 
        cnfg.DETAILED_COLS)
    print("Усреднение циклов завершено")

    df_avg_logs = df_avg_logs.astype({'Dia_Shell': int, 'L_Shell': int})
    df_avg_registry = df_avg_registry.astype({'Dia_Shell': int, 'L_Shell': int})  

    df_avg_logs = dm.apply_target_logic(
        df_avg_logs, 
        df_avg_registry, 
        df_reference_setpoint, 
        cnfg.CONFIG_GENERATE_TARGET_TANK
    )

    df_avg_logs = dm.apply_target_logic(
        df_avg_logs, 
        df_avg_registry, 
        df_reference_setpoint, 
        cnfg.CONFIG_GENERATE_TARGET_PIPE
    )
    print("Расчет целевых значений по тех карте завершен")

    dm.save_database(df_avg_logs, cnfg_p.LOGS_AVG_FILE)
    dm.save_database(df_avg_registry, cnfg_p.REGISTRY_AVG_FILE)
    
    return df_avg_logs, df_avg_registry, df_reference_setpoint

def load_processed_data():
    # Ячейка 4: Загрузка обработанных данных
    df_logs = dm.load_database(cnfg_p.LOGS_AVG_FILE)
    df_registry = dm.load_database(cnfg_p.REGISTRY_AVG_FILE)
    df_reference_setpoint = dm.load_database(cnfg_p.REFERENCE_SETPOINT_FILE)

    print('Данные загружены')
    
    return df_logs, df_registry, df_reference_setpoint