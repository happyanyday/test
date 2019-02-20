# -*- coding: utf-8 -*-
# ################################ #
# Describe: flag 同步
# Author: Pan Feng
# Create: 2018.10.22
# ################################ #

import ConfigParser
import commands
import os
import sys

import MySQLdb

reload(sys)
sys.setdefaultencoding('utf-8')

PRODUCT_NAME = "autohmpg"

# 是否打印日志
SHOWLOG = True


def p(*args):
    if SHOWLOG:
        try:
            print " ".join(args)
        except:
            print " ".join([str.format("%s" % __x) for __x in args])


def join_list(s):
    return "/".join(s)


def remove_slash(s):
    return s.replace("//", "/")


class Configure:
    configure = {}

    def __init__(self):
        self.__set_product_name()
        self.__init_path()

    def __set_product_name(self):
        self.product_name = PRODUCT_NAME
        p("[Configure] product_name = \"%s\"" % PRODUCT_NAME)

    def __init_names(self):
        for name in self.configure.keys():
            self.configure[name]["name"] = name

    def __init_path(self):
        self.__set_local_path()
        self.__set_go_path()

    def __set_local_path(self):
        self.__local_file_path_list = sys.argv[0].split("/")
        self.local_file_name = self.__local_file_path_list[-1]
        p("[Configure] local_file_name = \"%s\"" % self.local_file_name)

    def __set_go_path(self):
        status, output = commands.getstatusoutput('echo $GOPATH')
        self.__go_path = output


# 重写
class IniParser(ConfigParser.ConfigParser):
    def __init__(self, defaults=None):
        ConfigParser.ConfigParser.__init__(self, defaults=defaults)

    def optionxform(self, _optionstr):
        return _optionstr


class InitFlagFromSQL:
    def __init__(self, _conf):
        p("[InitYAMLFile] 当前目录 \"%s\"" % os.getcwd())
        self.__init_project_name(_conf)
        try:
            self.__connect()
            self.__get_result()
        except Exception, e:
            print "Get data from database error"
            print e

    def __init_project_name(self, _conf):
        if _conf.product_name == "autohmpg":
            self.product_name = "autohmpg2"
        else:
            self.product_name = _conf.product_name

    def __connect(self):
        __conn = MySQLdb.Connect(host='resources-pool-mw0-3306-db.sjz.autohome.com.cn',  # host
                                 db='dm_config',  # dbname
                                 port=3306,
                                 user='dm_config_wr',  # dbusername
                                 passwd='GkUEaHPLK*wSovM85&pa',  # dbpassword
                                 charset='utf8')
        __cur = __conn.cursor()
        __cur.execute("SELECT flagname,flagvalue FROM dm_flag_cfg WHERE projectname='%s'" % self.product_name)
        self.data = __cur.fetchall()
        __cur.close()
        __conn.close()

    def __get_result(self):
        self.__p = IniParser()
        self.__p.add_section('myFlag')
        self.__d = {}
        for row in self.data:
            key = str(row[0])
            if str(row[1]) == "1":
                value_str = "true"
            else:
                value_str = "false"
            self.__p.set('myFlag', key, value_str)
            self.__d[key] = value_str

        p("[InitFlagFromSQL] 获取SQL数据成功.")

    def write(self):
        with open("myflag.ini", "w+") as fw:
            self.__p.write(fw)
        p("[InitFlagFromSQL] 保存到文件\"myflag.ini\".")


if __name__ == "__main__":
    c = Configure()
    s = InitFlagFromSQL(c)
    s.write()
