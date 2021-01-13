import os

def implicit():
    from google.cloud import storage

    # If you don't specify credentials when constructing the client, the
    # client library will look for credentials in the environment.
    storage_client = storage.Client()   

    # Make an authenticated API request
    buckets = list(storage_client.list_buckets())
    print(buckets[1])
    gcp_bucket = "picture-classifier24902"
    data_dir = os.path.join("gs://",gcp_bucket,"pic")
    print(data_dir)
    print("hey"+"hey")
    print("gs://"+gcp_bucket+"/pic")
    
implicit()