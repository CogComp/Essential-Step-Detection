from aser.database.kg_connection import ASERKGConnection
from aser.extract.aser_extractor import SeedRuleASERExtractor, DiscourseASERExtractor

def extract_and_db_match(extractor, db, filein, fileout):

    fw = open(fileout, 'w')
    fread = open(filein)

    fread.readline()
    for line in fread:
        info = line.strip().split('\t')
        text = info[1]
        event, rel = extractor.extract_from_text(text, in_order=True)

        for event_ in event[0]:
            #db.get_partial_match_eventualities(event[0][0], bys=['words', 'verbs', "skeleton_words"], threshold=0.2)
            rets = db.get_exact_match_eventualities(event_)
            print('event_: ', event.__repr__())
            print(rets)

        break


if __name__ == "__main__":

    #aser_extractor = DiscourseASERExtractor(corenlp_path="stanford-corenlp-3.9.2", corenlp_port=9000)
    #db = ASERKGConnection(db_path='/home/data/corpora/aser/database/filter_2.0/100/KG.db', mode='memory')
    db = ASERKGConnection(db_path='/home/data/corpora/aser/database/filter_2.0/10/KG.db', mode='memory')
    #db = ASERKGConnection(db_path='/home/data/corpora/aser/database/filter_2.0/2/KG.db', mode='memory')

    #extract_and_db_match(aser_extractor, db, './dev.txt', './dev_match.txt')

    #extract eventuality
    #fw = open('/home/ysuay/codes/wino/preprocess/data/ASER_raw_data/aser_2.txt', 'w')
    fw = open('/home/ysuay/codes/wino/preprocess/data/ASER_raw_data/aser_10_tmp.txt', 'w')
    keys = db.eid2eventuality_cache.keys()
    for key in keys:
        fw.write('{}\t{}\n'.format(key, db.eid2eventuality_cache[key].__repr__()))
    fw.close()
    #db.close()

    #extract relation table

    #fw = open('/home/ysuay/codes/wino/preprocess/data/ASER_raw_data/aser_2_relation.txt', 'w')
    fw = open('/home/ysuay/codes/wino/preprocess/data/ASER_raw_data/aser_10_relation.txt', 'w')
    keys = db.rid2relation_cache.keys()
    for key in keys:
        relation = db.rid2relation_cache[key]
        fw.write('{}\t{}\t{}\n'.format(relation.hid, relation.tid, relation.relations))
    fw.close()
    #db.close()
    

