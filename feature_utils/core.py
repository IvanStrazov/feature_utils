# utf-8
# Python 3.10.0
# 2022-03-08


from typing import Optional, Iterable, Union, List, Tuple, Callable
import copy
import re
import functools

import pandas as pd
import numpy as np



####################################################################################################



class _TargetCategoryEncoderSpecialFunction:
    """
    Класс с определением кастомных функций.
    """
    
    _special_stats = dict()
    
    
    @staticmethod
    def _fun_quantile(stat: str) -> Callable:
        """
        Расчет квантиля.
        """
        
        q = float(stat.split("_")[1]) / 100
        fun = functools.partial(np.quantile, q=q)
        
        return fun
    
    
    @staticmethod
    def _fun_iqr(stat: str) -> Callable:
        """
        Расчет интерквартильного размаха.
        """
        
        _, *args = stat.split("_")
        
        if args:
            min_q, max_q = float(args[0])/100, float(args[1])/100
            fun = lambda a: np.quantile(a, max_q) - np.quantile(a, min_q)
        else:
            fun = lambda a: np.quantile(a, 0.75) - np.quantile(a, 0.25)
        
        return fun
    
    
    @staticmethod
    def _fun_diff(stat: str) -> Callable:
        """
        Расчет лагов.
        """
        
        _, diff_type, _n = stat.split("_")
        n = int(_n)
        
        if diff_type == "abs":
            fun = lambda x: x.diff(n)
        elif diff_type == "pct":
            fun = lambda x: x.pct_change(n)
        else:
            raise ValueError(f"wrong value of parameter stat' {stat}'")
        
        return fun



####################################################################################################



class _TargetCategoryEncoder(_TargetCategoryEncoderSpecialFunction):
    """
    Основной класс категориального кодирования таргета.
    """
    
    _default_stats = {"mean", "median", "min", "max", "count", "var", "std", "first", "last"}
   
    _classic_stats = {
        "classic": {"min", "q_25", "median", "q_75", "max", "mean", "std"},
        "classic_exp": {"min", "q_25", "median", "q_75", "max", "mean", "std"},
        "classic_ts": {"min", "q_25", "median", "q_75", "max", "mean", "std", "first", "last"}
    }
    
    _transform_stat_prefixes = {"diff"}
    
    
    def _is_transform_stat(self,
                           stat: str) -> bool:
        """
        Проверка является ли статистика "оконной трансформацией"
        """
        
        if stat.split("_")[0] in self._transform_stat_prefixes:
            return True
        else:
            return False
    
    
    def _stat_to_name_and_fun(self,
                              stat: str
                             ) -> Tuple[str]:
        """
        Определение интеграции отдельных категорий статистик.
        """
        
        if stat in self._default_stats:
            return (stat, stat)
        
        if stat in self._special_stats:
            return (stat, self._special_stats[stat])
        
        if re.match(r"q_\d{2}", stat):
            return (stat, self._fun_quantile(stat))
        
        if stat == "iqr" or re.match(r"iqr_\d+_\d+", stat):
            return (stat, self._fun_iqr(stat))
        
        if re.match(r"diff_(?:abs|pct)_\d+", stat):
            return (stat, self._fun_diff(stat))
        
        else:
            raise ValueError(f"wrong value of parameter stat '{stat}'")
    
    
    @staticmethod
    def _new_feat_names(categories: List[str],
                        target: str,
                        statistics: Union[str, List[str]]
                       ) -> Union[str, List[str]]:
        """
        Генерация списка названий новых признкаов.
        
        Параметры:
            categories (list[str]) - список колонок для группировки
            targets (list[str]) - список колонок с таргетом
            statistics (list[str]) - список статистик
        
        Возвращает:
            (str|list[str]) - список новых названий
        """
        
        if isinstance(statistics, tuple):
            f = lambda stat: "_".join(("tce", target, *categories, stat))
            return list(map(f, statistics))
        else:
            return "_".join(("tce", target, *categories, statistics))
    
    
    def _encode_agg(self,
                    data,
                    categories: List[str],
                    target: str,
                    stat_names: List[str],
                    stat_funs: List[Union[str, Callable]]
                   ):
        """
        Кодирование агрегатными статистиками.
        """
        
        data_agg = data.groupby(categories).agg({target: stat_funs})
        data_agg.columns = self._new_feat_names(categories, target, stat_names)
        
        return data.merge(data_agg, left_on=categories, right_index=True)
    
    
    def _encode_transform(self,
                          data,
                          categories: List[str],
                          target: str,
                          stat_names: List[str],
                          stat_funs: List[Union[str, Callable]]
                         ):
        """
        Кодирование оконными трансформациями.
        """
        
        for name, fun in zip(stat_names, stat_funs):
            new_name = self._new_feat_names(categories, target, name)
            data[new_name] = data.groupby(categories)[[target]].transform(fun)
        
        return data
    
    
    def target_cat_encode(self,
                          data,
                          categories: List[str],
                          targets: List[str],
                          statistics: List[str],
                          user_stats: dict = {},
                          sort_order: Optional[List[str]] = None,
                          return_inf: bool = False
                         ):
        """
        Кодирование категориальных признаков таргетом.
        
        Параметры:
            data (pandas DataFrame) - исходные данные
            categories (list[str]) - список колонок для группировки
            targets (list[str]) - список колонок с таргетом
            statistics (list[str]) - список статистик
            user_stats (dict, def={}) - словарь пользовательских статистик:
                key (str) - статистика
                value (function) - функция расчета статистики
            sort_order (list[str], def=None) - список колонок для сортировки
            return_inf (bool, def=False) - формат выходных данных
        
        Возвращает:
            data (pandas DataFrame) - выходные данные
        """
        
        _data = copy.deepcopy(data)
        
        # Распаковка обычных и "комбо" статистик в общий список
        _statistics = []
        for stat in statistics:
            if stat in self._classic_stats.keys():
                _statistics.extend(self._classic_stats[stat])
            else:
                _statistics.append(stat)
        _statistics = set(_statistics)
        
        # Обновление словаря "особых" статистик на пользовательские
        if user_stats is not None:
            self._special_stats.update(user_stats)
        
        # Сортировка
        if sort_order is not None:
            _data.sort_values(by=sort_order, inplace=True)
        
        # Обработка неагрегируемых статистик
        tr_stats = {stat for stat in _statistics if self._is_transform_stat(stat)}
        if tr_stats:
            tr_stat_names, tr_stat_funs = list(zip(*map(self._stat_to_name_and_fun, tr_stats)))
            for target in targets:
                _data = self._encode_transform(_data, categories, target, tr_stat_names, tr_stat_funs)
        
        # Обработка агрегируеых статистик
        agg_stats = _statistics.difference(tr_stats)
        if agg_stats:
            agg_stat_names, agg_stat_funs = list(zip(*map(self._stat_to_name_and_fun, agg_stats)))
            for target in targets:
                 _data = self._encode_agg(_data, categories, target, agg_stat_names, agg_stat_funs)
        
        # Определение формата возвращаемых значений
        if return_inf:
            return _data, agg_stat_names, tr_stat_names
        else:
            return _data



####################################################################################################



class FeatureMaker(_TargetCategoryEncoder):
    """
    Управляющий объект.
    """
    
    pass



####################################################################################################
