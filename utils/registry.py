#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. 

""" Registry class. """

class Registry(object):
    """
    The Registry class provides a registry for all things
    To initialize:
        REGISTRY = Registry()
    
    To register a tracker:
        @REGISTRY.register()
        class Model():
            ...
    """

    def __init__(self, table_name=""):
        """
        Initializes the registry.
        Args:
            table_name (str): specifies the name of the registry
        """
        self._entry_map = {}
        self.table_name = table_name


    def _register(self, name, entry):
        """
        Registers the instance.
        Args:
            name (str): name of the entry
            entry ():   instance of the entry, could be any type
        """
        assert type(name) is str
        assert (name not in self._entry_map.keys()), "{} {} already registered.".format(
            self.table_name, name
        )
        self._entry_map[name] = entry
    
    def register(self):
        """
        Wrapper function for registering a module.
        """
        def reg(obj):
            name = obj.__name__
            self._register(name, obj)
            return obj
        return reg
    
    def get(self, name):
        """
        Returns the instance specified by the name. 
        Args:
            name (str): name of the specified instance.
        """
        if name not in self._entry_map.keys():
            return None
        obj = self._entry_map.get(name)
        return obj
    
    def get_all_registered(self):
        """
        Prints all registered class. 
        """
        return self._entry_map.keys()