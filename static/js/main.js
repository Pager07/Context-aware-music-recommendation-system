$(document).ready(function(){
	var currentUserID = '29235188'
	$('#changeBttn').on('click' , function(event){
		$.ajax({
			type: 'POST',
			url: '/changeUser'
		}).done(function(data){
			$('#recommended_songs').empty().append(data.datax)
			$('#current_user').empty().text('Current user:' + data.currentUser);
			currentUserID = data.currentUser
		});
	});
	$('#getSongs').on('click',function(event){
		$.ajax({
			type: 'POST',
			url: '/getRecommendation'
		}).done(function(data){
			$('#recommended_songs').empty().append(data.datax)
		});
	});
});