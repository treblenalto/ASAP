import os
import time
import torch
import torch.optim as optim
from model import get_model
from data import load_data

def train(model, train_loader, optimizer, epoch, device = 'cuda'):
    model.train()                                        
    for batch_idx, (images, targets) in enumerate(train_loader):
        # data, target 값 DEVICE에 할당
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  

        optimizer.zero_grad()                 # optimizer gradient 값 초기화
        losses = model(images, targets)       # calculate loss

        loss = losses['loss_keypoint']        # keypoint loss
        loss.backward()                       # loss back propagatioin
        optimizer.step()                      # parameter update

        if (batch_idx+1) % 200 == 0:
            print(f'| epoch: {epoch} | batch: {batch_idx+1}/{len(train_loader)}')

def evaluate(model, test_loader, device = 'cuda'):
    model.train()      
    test_loss = 0      # test_loss 초기화
    
    with torch.no_grad(): 
        for images, targets in test_loader:
            # data, target 값 DEVICE에 할당
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  

            losses = model(images, targets)                       # validation loss
            test_loss += float(losses['loss_keypoint'])           # sum of all loss 
    
    test_loss /= len(test_loader.dataset)                         # 평균 loss
    return test_loss

def train_model(train_loader, val_loader, num_epochs = 30, device= 'cuda'):
    model = get_model()
    model.to(device)
    
    best_loss = 999999  # 가장 낮은 loss 초기화
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)

    for epoch in range(1, num_epochs+1):
        since = time.time()
        train(model, train_loader, optimizer, epoch, device)
        train_loss = evaluate(model, train_loader)
        val_loss = evaluate(model, val_loader)

        if val_loss <= best_loss:   # update best loss
          best_loss = val_loss
          torch.save(model, '../models/RCNN_ep'+str(epoch)+'_'+str(best_loss)+'.pt')
          print('Best Model Saved, Loss: ', val_loss)
        
        time_elapsed = time.time()-since
        print()
        print('---------------------- epoch {} ------------------------'.format(epoch))
        print('Train Keypoint Loss: {:.4f}, Val Keypoint Loss: {:.4f}'.format(train_loss, val_loss))   
        print('Completed in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
        print()

def main():
    path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(path)
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_img_path = '../images/images_1'
    train_key_path = './annotations/annotations_1.csv'

    train_loader, valid_loader = load_data(train_img_path, train_key_path)
    train_model(train_loader, valid_loader, num_epochs = 5, device = DEVICE) 
    '''
    default: epoch - 30, 
             device - cuda
    '''
if __name__=="__main__":
    main()