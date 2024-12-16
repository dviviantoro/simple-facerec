print("saya disini lho")

import redis

r = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=True
    )

def redis_clear():
    r.flushall()
    print("[OK] redis flush all data")

def redis_set(key, value):
    try:
        r.set(key, value)
        print(f"[OK] redis set ({key}, {value})")
    except Exception as e:
        print(f"[FAILED] redis set ({key}, {value})")
        print(e)

def redis_get(key):
    result = r.get(key)
    try:
        print(f"[OK] redis get ({key}, {result})")
        # return result.decode('utf-8')
        return result
    except:
        print(f"[FAILED] redis get ({key}, {result})")
        return False
    
def redis_pub(channel, value):
    try:
        r.publish(channel, value)
        print(f"[OK] redis publish ({channel}, {value})")
    except Exception as e:
        print(f"[FAILED] redis publish ({channel}, {value})")
        print(e)

def redis_sub(channel):
    print(f"[OK] redis subscribing {channel}")
    pubsub = r.pubsub()
    pubsub.subscribe(channel)

    for message in pubsub.listen():
        if message['type'] == 'message':
            print(f"[OK] redis subscribing result from {channel} : {message['data']}")
            return message['data']

def redis_sub_white():
    pubsub = r.pubsub()
    return pubsub

def redis_del(key):
    try:
        r.delete(key)
        print(f"[OK] redis delete {key}")
    except Exception as e:
        print(f"[FAILED] redis delete {key}")
        print(e)

def redis_clear_x():
    redis_del("invoice")
    redis_del("x_qr_str")
    redis_del("x_id")
    redis_del("x_status")
    redis_del("x_expired")

redis_pub("test", "redis terpubish")